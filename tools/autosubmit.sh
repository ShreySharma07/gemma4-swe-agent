#!/bin/bash
# Auto-submit one pre-approved zip as soon as the pending submission WAIT scores.
# Usage: autosubmit.sh WAIT_REF ZIP SHA16 "message"   (DRY=1 prints instead of submitting)
cd "$(dirname "$0")/.."
WAIT=$1; ZIP=$2; SHA=$3; MSG=$4
C=gemma-4-developer-agent
for i in $(seq 1 200); do  # up to ~33 h at 10 min
  line=$(kaggle competitions submissions $C 2>&1 | grep "^$WAIT ")
  st=$(echo "$line" | grep -o 'SubmissionStatus\.[A-Z]*')
  echo "$(date -u '+%m-%d %H:%M') wait $WAIT: ${st:-no status} | $(echo "$line" | awk '{print $NF}')"
  case "$st" in *COMPLETE*|*ERROR*) break;; esac
  [ -n "$DRY" ] && { echo "DRY: would keep waiting"; exit 0; }
  sleep 600
done
case "$st" in *COMPLETE*|*ERROR*) ;; *) echo "GAVE UP waiting"; exit 2;; esac
[ "$(shasum -a 256 "$ZIP" | cut -c1-16)" = "$SHA" ] || { echo "ZIP CHANGED; not submitting"; exit 1; }
for t in $(seq 1 30); do
  pend=$(kaggle competitions submissions $C 2>&1 | grep -c 'SubmissionStatus.PENDING')
  if [ "$pend" != "0" ]; then echo "$(date -u +%H:%M) a submission is pending; waiting"; [ -n "$DRY" ] && exit 0; sleep 600; continue; fi
  if [ -n "$DRY" ]; then echo "DRY: would submit $ZIP ($SHA) -m \"$MSG\""; exit 0; fi
  r=$(kaggle competitions submit -c $C -f "$ZIP" -m "$MSG" 2>&1)
  echo "$(date -u '+%m-%d %H:%M:%S') $(echo "$r" | tail -3 | tr '\n' ' ')"
  case "$r" in *"Successfully submitted"*) echo SUBMITTED; exit 0;; esac
  sleep 600
done
echo "GAVE UP submitting"; exit 3
