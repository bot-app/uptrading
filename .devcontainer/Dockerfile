FROM khulnasoft/uptrading:develop_upairl

USER root
# Install dependencies
COPY requirements-dev.txt /uptrading/

RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog \
    && apt-get -y install --no-install-recommends git sudo vim build-essential \
    && apt-get clean \
    && mkdir -p /home/ftuser/.vscode-server /home/ftuser/.vscode-server-insiders /home/ftuser/commandhistory \
    && echo "export PROMPT_COMMAND='history -a'" >> /home/ftuser/.bashrc \
    && echo "export HISTFILE=~/commandhistory/.bash_history" >> /home/ftuser/.bashrc \
    && chown ftuser:ftuser -R /home/ftuser/.local/ \
    && chown ftuser: -R /home/ftuser/

USER ftuser

RUN pip install --user autopep8 -r docs/requirements-docs.txt -r requirements-dev.txt --no-cache-dir

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
