# Dockerfile for frozen-intelligence
# Zero external dependencies — Python stdlib only

FROM python:3.11-slim

WORKDIR /app

# Copy source
COPY src/ ./src/
COPY tests/ ./tests/
COPY cli.py .
COPY README.md .

# Create entrypoint
RUN echo '#!/bin/bash\ncd /app\npython3 cli.py "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Test that it works
RUN python3 -c "import sys; sys.path.insert(0, 'src'); from tlmm_engine import TLMMConfig; print('TLMM engine imported')"

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
