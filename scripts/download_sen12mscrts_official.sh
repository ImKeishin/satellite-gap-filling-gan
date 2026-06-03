#!/usr/bin/env bash
set -euo pipefail

# Official helper script maintained by the SEN12MS-CR-TS authors.
# It interactively asks for region, whether S1 is needed, and destination.
wget -c -O dl_data.sh \
  https://raw.githubusercontent.com/PatrickTUM/SEN12MS-CR-TS/master/util/dl_data.sh
chmod +x dl_data.sh
./dl_data.sh
