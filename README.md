# Alpaca
This is Eric's gigantic Hive project.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

## Overview
Eric's work is under Alpaca.
The intern's team work is under Astra.

### Requirements
* Docker-CE version 17.06.2-ce
* Docker Compose version 1.14.0

### Getting Started
To install, hop into Docker and install the necessary datasets.
```
docker-compose up --build
```

Now hop into Docker and download some files.
```
cd Alpaca
bash access_Alpaca.sh
```

Now you should be inside the Alpaca container.
```
service neo4j start
python3 -m Alpaca.datasets.generic_double_seq.download
python3 -m Alpaca.datasets.generic_flat.download
python3 -m Alpaca.datasets.generic_sequential.download
```

Go into Neo4j by visting 0.0.0.0:7474 on your local browser.
Run the following queries: 
```
CREATE INDEX ON :Resource(uri)
CALL semantics.importRDF('file:///Alpaca/datasets/apple.owl','RDF/XML', {})
```

Now run unit tests to make sure everything is awesome.
```
py.test Alpaca/tests
```

### Contribute
I appreciate all contributions. Just make a pull request.
Contributors are listed under `contributors.txt`.

## License
MIT License

Copyright (c) 2017 Eric Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

