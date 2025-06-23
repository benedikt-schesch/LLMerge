#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reads the combined evaluation results, selects 20 "mistakes" for each model
by maximizing the overlap between selections, and outputs the result in a
one-row-per-model CSV format.
"""

import pandas as pd

# Constants for file paths, adhering to UPPER_CASE naming style
COMBINED_RESULTS_FILE = "tables/all_results_combined.csv"
FINAL_OUTPUT_FILE = "tables/mistake_analysis_per_model.csv"
NUM_MISTAKES_TO_SELECT = 20


def _read_and_identify_mistakes(df: pd.DataFrame):
    """Identifies mistakes from the DataFrame."""
    return df[(~df["semantically_resolved"]) & (df["has_valid_java_md"])].copy()


def _prioritize_mistakes(mistakes_df: pd.DataFrame):
    """Calculates the frequency of each mistake to determine priority."""
    mistake_priority = mistakes_df["index"].value_counts().reset_index()
    mistake_priority.columns = ["index", "priority"]
    return mistake_priority.sort_values(by="priority", ascending=False)


def _perform_greedy_selection(
    sorted_mistakes: pd.DataFrame,
    mistake_to_models: dict,
    selections: dict,
    num_mistakes: int,
):
    """Greedily selects mistakes for each model based on priority."""
    for _, row in sorted_mistakes.iterrows():
        merge_index = row["index"]
        failing_models = mistake_to_models.get(merge_index, [])
        for model in failing_models:
            if (
                len(selections[model]) < num_mistakes
                and merge_index not in selections[model]
            ):
                selections[model].append(merge_index)
    return selections


def _fill_remaining_mistakes(
    selections: dict, mistakes_df: pd.DataFrame, all_models: list, num_mistakes: int
):
    """Fills in any remaining slots for models with fewer than the desired number of mistakes."""
    for model in all_models:
        num_needed = num_mistakes - len(selections[model])
        if num_needed > 0:
            model_specific_mistakes = set(
                mistakes_df[mistakes_df["model"] == model]["index"]
            )
            available_to_add = list(model_specific_mistakes - set(selections[model]))
            selections[model].extend(available_to_add[:num_needed])

            if len(available_to_add) < num_needed:
                # The long line is broken into a multi-line f-string
                warning_message = (
                    f"Warning: Model '{model}' has only "
                    f"{len(model_specific_mistakes)} total mistakes. "
                    f"Could not select {num_mistakes}."
                )
                print(warning_message)
    return selections


def analyze_and_select_mistakes(
    input_csv: str, output_csv: str, num_mistakes: int = 20
):
    """
    Reads combined results, selects mistakes, and saves them in a wide format.

    A "mistake" is defined as a merge where 'semantically_resolved' is False
    but 'has_valid_java_md' is True.
    """
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv}'")
        return

    mistakes_df = _read_and_identify_mistakes(df)
    if mistakes_df.empty:
        print("No mistakes found according to the definition.")
        return

    sorted_unique_mistakes = _prioritize_mistakes(mistakes_df)
    mistake_to_models = mistakes_df.groupby("index")["model"].apply(list).to_dict()

    all_models = df["model"].unique()
    selections = {model: [] for model in all_models}

    selections = _perform_greedy_selection(
        sorted_unique_mistakes, mistake_to_models, selections, num_mistakes
    )
    selections = _fill_remaining_mistakes(
        selections, mistakes_df, all_models, num_mistakes
    )

    output_data = []
    for model, indices in selections.items():
        if indices:
            indices_str = ",".join(map(str, sorted(indices)))
            output_data.append({"model": model, "selected_indices": indices_str})

    pd.DataFrame(output_data).to_csv(output_csv, index=False)

    print("Analysis complete.")
    print(f"Total unique mistakes identified: {len(sorted_unique_mistakes)}")
    print(f"Final aggregated mistake list saved to '{output_csv}'")


if __name__ == "__main__":
    # This script assumes 'run_evaluations.sh' has already been run
    analyze_and_select_mistakes(
        COMBINED_RESULTS_FILE, FINAL_OUTPUT_FILE, NUM_MISTAKES_TO_SELECT
    )
