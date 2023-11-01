ARG sourceimage=khulnasoft/uptrading
ARG sourcetag=develop
FROM ${sourceimage}:${sourcetag}

# Install dependencies
COPY requirements-plot.txt /uptrading/

RUN pip install -r requirements-plot.txt --user --no-cache-dir
