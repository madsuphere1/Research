#!/usr/bin/env bash
# Re-clone the third-party reference repos into external/ (gitignored).
# Notes:
#  * twopirllc/pandas-ta went private -> use the community fork pandas-ta-classic.
#  * ElliottWaveAnalyzer is no longer public -> the `taew` PyPI package is used instead.
set -euo pipefail
cd "$(dirname "$0")/../external"

repos=(
  https://github.com/joshyattridge/smart-money-concepts      # SMC / market structure
  https://github.com/xgboosted/pandas-ta-classic             # TA indicators (pandas-ta fork)
  https://github.com/white07S/TradingPatternScanner          # classical chart patterns
  https://github.com/niall-oc/pyharmonics                    # harmonic patterns
  https://github.com/bfolkens/py-market-profile              # volume profile
  https://github.com/wilsonfreitas/awesome-quant             # curated quant index
  https://github.com/stefan-jansen/machine-learning-for-trading  # ML4T book code
  https://github.com/cantaro86/Financial-Models-Numerical-Methods # pricing / numerical methods
)

for r in "${repos[@]}"; do
  name=$(basename "$r")
  [ -d "$name" ] || git clone --depth 1 "$r"
done
