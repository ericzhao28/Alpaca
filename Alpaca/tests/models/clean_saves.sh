rm -rf saves/double_seq/checkpoints/*
rm -rf saves/double_seq/graphs/test/*
rm -rf saves/double_seq/graphs/train/*

rm -rf saves/flat/checkpoints/*
rm -rf saves/flat/graphs/test/*
rm -rf saves/flat/graphs/train/*

rm -rf saves/sequential/checkpoints/*
rm -rf saves/sequential/graphs/test/*
rm -rf saves/sequential/graphs/train/*

touch saves/double_seq/checkpoints/.gitkeep
touch saves/double_seq/graphs/test/.gitkeep
touch saves/double_seq/graphs/train/.gitkeep

touch saves/flat/checkpoints/.gitkeep
touch saves/flat/graphs/test/.gitkeep
touch saves/flat/graphs/train/.gitkeep

touch saves/sequential/checkpoints/.gitkeep
touch saves/sequential/graphs/test/.gitkeep
touch saves/sequential/graphs/train/.gitkeep

