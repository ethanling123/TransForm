#!/bin/sh
# Shape this machine's outgoing traffic with the kernel's netem, then run the command. Each machine delays and caps what
# it sends, so a round trip costs twice NETEM_DELAY_MS and both directions get the rate cap.
#   NETEM_DELAY_MS, NETEM_RATE_MBIT, NETEM_LOSS_PCT   (all unset = an unshaped link)
set -e
if [ -n "$NETEM_DELAY_MS" ]; then
  args="delay ${NETEM_DELAY_MS}ms limit 100000"
  [ -n "$NETEM_LOSS_PCT" ] && args="$args loss ${NETEM_LOSS_PCT}%"
  [ -n "$NETEM_RATE_MBIT" ] && args="$args rate ${NETEM_RATE_MBIT}mbit"
  tc qdisc add dev eth0 root netem $args
  echo "netem on $(hostname): $args"
fi
exec "$@"
