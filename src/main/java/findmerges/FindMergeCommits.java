package findmerges;

import com.opencsv.CSVReaderHeaderAware;
import com.opencsv.exceptions.CsvValidationException;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.ListBranchCommand;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.internal.storage.file.FileRepository;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.merge.RecursiveMerger;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;
import org.eclipse.jgit.revwalk.filter.RevFilter;
import org.eclipse.jgit.transport.CredentialsProvider;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;
import org.kohsuke.github.GitHub;
import org.kohsuke.github.GitHubBuilder;
import org.plumelib.util.StringsPlume;
import me.tongfei.progressbar.ProgressBar;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Given a list of repositories, outputs a list of merge commits. The merge commits may be on the
 * mainline branch, feature branches, and pull requests (both opened and closed).
 *
 * <p>The input is a .csv file, one of whose columns is named "repository" and contains "org/repo".
 *
 * <p>The output is a set of {@code .csv} files with columns: branch name, merge commit SHA, parent
 * 1 commit SHA, parent 2 commit SHA, base commit SHA, notes. The "notes" column contains "a parent
 * is the base", "two initial commits", or is blank.
 *
 * <p>Requires (because JGit requires authentication for cloning and fetching public repositories):
 *
 * <ul>
 *   <li>the existence of a {@code GITHUB_TOKEN} environment variable (GitHub Actions provides
 *       this), or
 *   <li>a {@code .github-personal-access-token} file in your home directory whose first line is
 *       your GitHub username, whose second line is a read-only personal access token, and all other
 *       lines are ignored.
 * </ul>
 */
public class FindMergeCommits {

  // new total merge cap and fixed random seed for reproducibility
  private static final int MAX_TOTAL_MERGE_COMMITS = 10000;
  private static final long RANDOM_SEED = 42;

  /** The GitHub repositories to search for merge commits. */
  final List<OrgAndRepo> repos;

  /** The output directory. */
  final Path outputDir;

  /** Performs GitHub queries and actions. */
  final GitHub gitHub;

  /** The JGit credentials provider. */
  final CredentialsProvider credentialsProvider;
  
  /**
   * Outputs a list of merge commits from the given repositories.
   *
   * @param args the first element is a .csv file containing GitHub repository slugs, in "org/repo"
   *     format, in a column named "repository"; the second element is an output directory
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  public static void main(String[] args) throws IOException, GitAPIException {
    if (args.length != 2) {
      System.err.printf("Usage: FindMergeCommits <repo-csv-file> <output-dir>%n");
      System.exit(1);
    }

    String inputFileName = args[0];
    List<OrgAndRepo> repos = reposFromCsv(inputFileName);

    String outputDirectoryName = args[1];
    Path outputDir = Paths.get(outputDirectoryName);

    FindMergeCommits fmc = new FindMergeCommits(repos, outputDir);

    fmc.writeMergeCommitsForRepos();
  }

  /**
   * Creates an instance of FindMergeCommits.
   *
   * @param repos a list of GitHub repositories
   * @param outputDir where to write results; is created if it does not exist
   * @throws IOException if there is trouble reading or writing files
   */
  FindMergeCommits(List<OrgAndRepo> repos, Path outputDir) throws IOException {
    this.repos = repos;
    this.outputDir = outputDir;

    this.gitHub =
        GitHubBuilder.fromEnvironment()
            .build();

    outputDir.toFile().mkdirs();

    File tokenFile = new File(System.getProperty("user.home"), ".github-personal-access-token");
    String environmentGithubToken = System.getenv("GITHUB_TOKEN");

    String gitHubUsername;
    String gitHubPersonalAccessToken;
    if (tokenFile.exists()) {
      try (
          @SuppressWarnings("DefaultCharset")
          BufferedReader pwReader = new BufferedReader(new FileReader(tokenFile))) {
        gitHubUsername = pwReader.readLine();
        gitHubPersonalAccessToken = pwReader.readLine();
      }
      if (gitHubUsername == null || gitHubPersonalAccessToken == null) {
        System.err.println("File .github-personal-access-token does not contain two lines.");
        System.exit(2);
      }
      this.credentialsProvider =
          new UsernamePasswordCredentialsProvider(gitHubUsername, gitHubPersonalAccessToken);
    } else if (environmentGithubToken != null) {
      this.credentialsProvider =
          new UsernamePasswordCredentialsProvider("Bearer", environmentGithubToken);
    } else {
      System.err.println(
          "FindMergeCommits: "
              + "need .github-personal-access-token file or GITHUB_TOKEN environment variable.");
      System.exit(3);
      throw new Error("unreachable"); // needed due to javac definite assignment check
    }
  }

  @Override
  public String toString(@GuardSatisfied FindMergeCommits this) {
    return String.format("FindMergeCommits(%s, %s)", repos, outputDir);
  }

  /** Represents a GitHub repository. */
  static class OrgAndRepo {
    /** The owner or organization. */
    public final String org;

    /** The repository name within the organization. */
    public final String repo;

    /**
     * Creates a new OrgAndRepo.
     *
     * @param org the org or organization
     * @param repo the repository name within the organization
     */
    public OrgAndRepo(String org, String repo) {
      this.org = org;
      this.repo = repo;
    }

    /**
     * Creates a new OrgAndRepo.
     *
     * @param orgAndRepoString the organization and repository name, separated by a slash ("/")
     */
    public OrgAndRepo(String orgAndRepoString) {
      String[] orgAndRepoSplit = orgAndRepoString.split("/", -1);
      if (orgAndRepoSplit.length != 2) {
        System.err.printf("repo \"%s\" has wrong number of slashes%n", orgAndRepoString);
        System.exit(4);
      }
      this.org = orgAndRepoSplit[0];
      this.repo = orgAndRepoSplit[1];
    }

    /**
     * Returns the printed representation of this.
     *
     * @return the printed representation of this
     */
    @Override
    public String toString(@GuardSatisfied OrgAndRepo this) {
      return org + "/" + repo;
    }
  }

  /**
   * Reads a list of repositories from a .csv file, one of whose columns is "repository".
   *
   * @param inputFileName the name of the input .csv file
   * @return a list of repositories
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  static List<OrgAndRepo> reposFromCsv(String inputFileName) throws IOException, GitAPIException {
    List<OrgAndRepo> repos = new ArrayList<>();
    try (
        @SuppressWarnings("DefaultCharset")
        FileReader fr = new FileReader(inputFileName);
        CSVReaderHeaderAware csvReader = new CSVReaderHeaderAware(fr)) {
      String[] repoColumn;
      while ((repoColumn = csvReader.readNext("repository")) != null) {
        assert repoColumn.length == 1 : "@AssumeAssertion(index): application-specific property";
        repos.add(new OrgAndRepo(repoColumn[0]));
      }
    } catch (CsvValidationException e) {
      throw new Error(e);
    }
    return repos;
  }

  /**
   * The main entry point of the class, which is called by {@link #main}.
   *
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  void writeMergeCommitsForRepos() throws IOException, GitAPIException {
    System.out.printf("FindMergeCommits: %d repositories.%n", repos.size());
    try (ProgressBar pb = new ProgressBar("Processing repos", repos.size())) {
      repos.parallelStream().forEach(repo -> {
          writeMergeCommitsForRepo(repo);
          pb.step();
      });
    }
  }

  /**
   * Writes all merge commits for the given repository to a file.
   *
   * @param orgAndRepo the GitHub organization name and repository name
   */
  void writeMergeCommitsForRepo(OrgAndRepo orgAndRepo) {
    String msgPrefix = StringsPlume.rpad("FindMergeCommits: " + orgAndRepo + " ", 69) + " ";
    System.out.println(msgPrefix + "STARTED");
    try {
      writeMergeCommits(orgAndRepo);
    } catch (Throwable e) {
      throw new Error(e);
    }
    System.out.println(msgPrefix + "DONE");
  }

  /**
   * Writes all merge commits for the given repository to a file.
   *
   * @param orgAndRepo the GitHub organization name and repository name
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  void writeMergeCommits(OrgAndRepo orgAndRepo) throws IOException, GitAPIException {
    String orgName = orgAndRepo.org;
    String repoName = orgAndRepo.repo;
    Path orgOutputDir = Paths.get(this.outputDir.toString(), orgName);
    orgOutputDir.toFile().mkdirs();
    File outputFile = new File(orgOutputDir.toFile(), repoName + ".csv");
    Path outputPath = outputFile.toPath();
    
    // Check if the file exists and if it has enough entries
    AtomicInteger idx = new AtomicInteger(1);
    boolean fileExists = Files.exists(outputPath);
    
    if (fileExists) {
      try (Stream<String> lines = Files.lines(outputPath)) {
        long lineCount = lines.count();
        // Header line doesn't count as a merge entry
        if (lineCount > 1) {
          int existingEntries = (int)lineCount - 1;
          if (existingEntries >= MAX_TOTAL_MERGE_COMMITS) {
            System.out.println("Output file already has " + existingEntries + " entries, which is enough. Skipping " + orgAndRepo);
            return;
          }
          System.out.println("Output file has " + existingEntries + " entries, but need " + MAX_TOTAL_MERGE_COMMITS + 
                            ". Computing more for " + orgAndRepo);
          // Set the index to continue from the last entry
          idx.set(existingEntries + 1);
        }
      }
    }

    String repoDirName =
        System.getenv().getOrDefault("REPOS_PATH", "repos")
            + "/"
            + orgName
            + "/"
            + repoName;
    File repoDirFile = new File(repoDirName);
    repoDirFile.mkdirs();

    Git git;
    try {
      git =
          Git.cloneRepository()
              .setURI("https://github.com/" + orgName + "/" + repoName + ".git")
              .setDirectory(repoDirFile)
              .setCloneAllBranches(true)
              .setCredentialsProvider(credentialsProvider)
              .call();
    } catch (Exception e) {
      System.out.println("Exception in cloning");
      try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
        writer.write("branch_name,merge_commit,parent_1,parent_2,notes");
        writer.newLine();
      }
      return;
    }
    FileRepository repo = new FileRepository(repoDirFile);

    makeBranchesForPullRequests(git);
    
    try (BufferedWriter writer = fileExists ? 
         Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8, 
                                java.nio.file.StandardOpenOption.APPEND) : 
         Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
         
      // Write the CSV header only for new files
      if (!fileExists) {
        writer.write("branch_name,merge_commit,parent_1,parent_2,notes");
        writer.newLine();
      }

      writeMergeCommitsForBranches(git, repo, orgName, repoName, writer, idx);
    }
  }

  /**
   * Write, to {@code writer}, all the merge commits in all the branches of the given repository,
   * stopping once we have collected {@value #MAX_TOTAL_MERGE_COMMITS}.
   *
   * @param git the JGit porcelain
   * @param repo the JGit file system repository
   * @param orgName the organization (owner) name
   * @param repoName the repository name
   * @param writer where to write the merge commits
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  void writeMergeCommitsForBranches(
      Git git, FileRepository repo, String orgName, String repoName, BufferedWriter writer, AtomicInteger idx)
      throws IOException, GitAPIException {

    // Create a unique key for this repo to use in caching
    String repoKey = orgName + "/" + repoName;
    
    // Cache for branches in this repository
    Map<String, List<Ref>> branchCache = new HashMap<>();
    
    List<Ref> branches;
    // Check if we have cached branches for this repo
    if (branchCache.containsKey(repoKey)) {
      branches = branchCache.get(repoKey);
      System.out.println("Using cached branches for repository " + repoKey);
    } else {
      branches = git.branchList().setListMode(ListBranchCommand.ListMode.ALL).call();
      
      // Sort branches deterministically by name before removing duplicates
      // to ensure the same branch is kept regardless of original order
      branches.sort(Comparator.comparing(Ref::getName));
      
      branches = withoutDuplicateBranches(branches);
      
      // Cache the branches for future use
      branchCache.put(repoKey, branches);
      
      System.out.println("Found and cached " + branches.size() + " unique branches in repository " + repoKey);
    }

    // Cache for merge commits per branch
    Map<String, List<RevCommit>> mergeCommitCache = new HashMap<>();
    
    // No parallel streaming; track total merges so far and stop after max merge commits
    int mergesSoFar = 0;
    Random random = new Random(RANDOM_SEED);
    int branchCount = 0;

    for (Ref branch : branches) {
      branchCount++;
      if (mergesSoFar >= MAX_TOTAL_MERGE_COMMITS) {
        System.out.println("Reached max merge commit limit of " + MAX_TOTAL_MERGE_COMMITS + ", stopping branch processing");
        break;
      }
      System.out.println("Processing branch " + branchCount + "/" + branches.size() + ": " + branch.getName());
      mergesSoFar = writeMergeCommitsForBranch(
          git, repo, branch, writer, idx, mergesSoFar, random, mergeCommitCache);
    }
  }

  /**
   * Write, to {@code writer}, all the merge commits in one branch of the given repository,
   * without exceeding {@value #MAX_TOTAL_MERGE_COMMITS}.
   *
   * @param git the JGit porcelain
   * @param repo the JGit file system repository
   * @param branch the branch whose commits to output
   * @param writer where to write the merges
   * @param idx atomic counter for numbering merges
   * @param mergesSoFar how many merges have been written so far (across all branches)
   * @param random the Random instance (seeded) for reproducibility
   * @param mergeCommitCache cache of merge commits per branch
   * @return updated mergesSoFar
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  int writeMergeCommitsForBranch(
      Git git,
      FileRepository repo,
      Ref branch,
      BufferedWriter writer,
      AtomicInteger idx,
      int mergesSoFar,
      Random random,
      Map<String, List<RevCommit>> mergeCommitCache)
      throws IOException, GitAPIException {

    ObjectId branchId = branch.getObjectId();
    if (branchId == null) {
      throw new Error("no ObjectId for " + branch);
    }

    // Create a unique key for this branch
    String branchKey = branch.getName();

    // Collect merges from this branch, using cache if available
    List<RevCommit> mergeCommits;
    if (mergeCommitCache.containsKey(branchKey)) {
      mergeCommits = mergeCommitCache.get(branchKey);
      System.out.println("Using cached merge commits for branch " + branchKey);
    } else {
      mergeCommits = new ArrayList<>();
      Iterable<RevCommit> commits = git.log().add(branchId).call();
      for (RevCommit commit : commits) {
        if (commit.getParentCount() == 2) {
          mergeCommits.add(commit);
        }
      }

      // Sort the list deterministically before shuffling to ensure consistent results
      // across different runs with the same seed, regardless of initial collection order
      mergeCommits.sort(Comparator.comparing(commit -> commit.getId().getName()));
      
      // Cache the merge commits for this branch
      mergeCommitCache.put(branchKey, new ArrayList<>(mergeCommits));
      System.out.println("Cached " + mergeCommits.size() + " merge commits for branch " + branchKey);
    }
    
    // If mergesSoFar + mergesInBranch exceeds the cap, randomly select only enough merges.
    // The random selection is still deterministic because we use a fixed seed
    int maxAllowedForThisBranch = MAX_TOTAL_MERGE_COMMITS - mergesSoFar;
    if (mergeCommits.size() > maxAllowedForThisBranch) {
      // Create a copy of the cached list before shuffling to preserve the cache
      mergeCommits = new ArrayList<>(mergeCommits);
      Collections.shuffle(mergeCommits, random);
      mergeCommits = mergeCommits.subList(0, maxAllowedForThisBranch);
    }

    // Keep track of merges we've written for this branch
    int mergesWrittenHere = 0;

    // Write them out
    int processedCount = 0;
    int totalToProcess = mergeCommits.size();

    for (RevCommit commit : mergeCommits) {
      processedCount++;
      // Double-check it's a merge
      if (commit.getParentCount() != 2) {
        continue;
      }
      ObjectId mergeId = commit.toObjectId();

      RevCommit parent1 = commit.getParent(0);
      RevCommit parent2 = commit.getParent(1);
      if (parent1.equals(parent2)) {
        continue;
      }

      ObjectId parent1Id = parent1.toObjectId();
      ObjectId parent2Id = parent2.toObjectId();
      RevCommit mergeBase = getMergeBaseCommit(git, repo, parent1, parent2);
      ObjectId mergeBaseId;
      String notes;

      if (mergeBase == null) {
        notes = "two initial commits";
        mergeBaseId = null;
      } else {
        mergeBaseId = mergeBase.toObjectId();
        if (mergeBaseId.equals(parent1Id) || mergeBaseId.equals(parent2Id)) {
          notes = "a parent is the base";
        } else {
          notes = "";
        }
      }

      // branch_name,merge_commit,parent_1,parent_2,notes
      String line = String.format(
          "%s,%s,%s,%s,%s",
          branch.getName(),
          ObjectId.toString(mergeId),
          ObjectId.toString(parent1Id),
          ObjectId.toString(parent2Id),
          notes);

      writer.write(String.format("%d,%s", idx.getAndIncrement(), line));
      writer.newLine();

      System.out.printf("  Merge %s: %d/%d\n", repo.getDirectory().getName(), processedCount, totalToProcess);

      mergesWrittenHere++;
    }

    return mergesSoFar + mergesWrittenHere;
  }

  /**
   * For each remote pull request branch, make a local branch.
   *
   * @param git the JGit porcelain
   * @throws IOException if there is trouble reading or writing files
   * @throws GitAPIException if there is trouble running Git commands
   */
  void makeBranchesForPullRequests(Git git) throws IOException, GitAPIException {
    git.fetch()
        .setRemote("origin")
        .setRefSpecs("refs/pull/*/head:refs/remotes/origin/pull/*")
        .call();
  }

  /// Git utilities

  /**
   * Returns a list, retaining only the first branch when multiple branches have the same head SHA,
   * such as refs/heads/master and refs/remotes/origin/master. The result list has elements in the
   * same order as the argument list.
   *
   * <p>This method removes duplicate branches. It does not filter duplicate commits that appear in
   * multiple branches.
   *
   * @param branches a list of branches
   * @return the list, with duplicates removed
   */
  @SuppressWarnings("nullness:methodref.return") // method reference, inference failed; likely #979
  List<Ref> withoutDuplicateBranches(List<Ref> branches) {
    return new ArrayList<>(
        branches.stream()
            .collect(Collectors.toMap(Ref::getObjectId, p -> p, (p, q) -> p, LinkedHashMap::new))
            .values());
  }

  /**
   * Given two commits, return their merge base commit. It is the nearest ancestor of both commits.
   * If there is none (because the two commits have different initial commits!), then this returns
   * null.
   *
   * <p>This always returns an existing commit (or null), never a synthetic one.
   *
   * <p>When a criss-cross merge exists in the history, this outputs an arbitrary one of the best
   * merge bases (likely the earliest one). It doesn't matter which one is output, for the uses of
   * this method in this program.
   *
   * @param git the JGit porcelain
   * @param repo the JGit repository
   * @param commit1 the first parent commit
   * @param commit2 the second parent commit
   * @return the merge base of the two commits, or null if none exists
   */
  @Nullable
  RevCommit getMergeBaseCommit(
      Git git, Repository repo, RevCommit commit1, RevCommit commit2) {
    if (commit1.equals(commit2)) {
      throw new Error(
          String.format(
              "Same commit passed twice: getMergeBaseCommit(%s, \"%s\", \"%s\")",
              repo, commit1, commit2));
    }

    try {
      List<RevCommit> history1 = new ArrayList<>();
      git.log().add(commit1).call().forEach(history1::add);
      List<RevCommit> history2 = new ArrayList<>();
      git.log().add(commit2).call().forEach(history2::add);

      if (history1.contains(commit2)) {
        return commit2;
      } else if (history2.contains(commit1)) {
        return commit1;
      }

      Collections.reverse(history1);
      Collections.reverse(history2);
      int minLength = Math.min(history1.size(), history2.size());
      int commonPrefixLength = -1;
      for (int i = 0; i < minLength; i++) {
        if (!history1.get(i).equals(history2.get(i))) {
          commonPrefixLength = i;
          break;
        }
      }

      if (commonPrefixLength == 0) {
        return null;
      } else if (commonPrefixLength == -1) {
        throw new Error(
            String.format(
                "Histories are equal for getMergeBaseCommit(%s, \"%s\", \"%s\")",
                repo, commit1, commit2));
      }

      return history1.get(commonPrefixLength - 1);
    } catch (Exception e) {
      throw new Error(
          String.format("getMergeBaseCommit(%s, %s, %s, %s)", git, repo, commit1, commit2), e);
    }
  }
}
