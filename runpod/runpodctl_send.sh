
$DIR=$1

echo "Syncing from local ($DIR) to runpod..."
runpodctl send $DIR
echo "Sync completed successfully."
