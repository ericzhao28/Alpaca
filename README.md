# No Need to Pay Attention
Implementation of No Need to Pay Attention: using recurrent networks without attention mechanisms to answer simple questions about a knowledge base.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

Paper: https://arxiv.org/pdf/1606.05029.pdf
Paper authors: Ferhan Ture and Oliver Jojic; Comcast Labs, Washington, DC 20005
Abstract:
```
"First-order factoid question answering assumes that the question can be answered by a single fact in a knowledge base (KB). While this does not seem like a challenging task, many recent attempts that apply either complex linguistic reasoning or deep neural networks achieve 65%–76% accuracy on benchmark sets. Our approach formulates the task as two machine learning problems: detecting the entities in the question, and classifying the question as one of the relation types in the KB. We train a recurrent neural network to solve each problem. On the SimpleQuestions dataset, our approach yields substantial improvements over previously published results — even neural networks based on much more complex architectures. The simplicity of our approach also has practical advantages, such as efficiency and modularity, that are valuable especially in an industry setting. In fact, we present a preliminary analysis of the performance of our model on real queries from Comcast’s X1 entertainment platform with millions of users every day."
```

### Dependencies
* Docker-CE Version 17.06.0-ce-mac19
* Python 3.6.1

### Installation
```
docker-compose up --build
bash access_cluster.sh
```

### Contributors
Contributors are listed under contributors.txt
Got suggestions? Make a pull request.

### License
Copyright 2017 Eric Zhao and Hive Data

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
