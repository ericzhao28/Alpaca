find . -name "__pycache__" -exec rm -rf {} \;
find . -name ".cache" -exec rm -rf {} \;
find . -name ".DS_Store" -exec rm -rf {} \;
find . -name "*.pyc" -delete
find . -name "*.swo" -exec rm -rf {} \;
find . -name "*.swp" -exec rm -rf {} \;
echo "Success: python cache files and vim checkpoints have been deleted"
