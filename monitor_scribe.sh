#!/bin/bash
# Self-terminating monitor: polls for the existence of the corpus and book
while true; do
    if [ -f "/home/adam/worxpace/gladius/raven/corpus/ARC_GOLDEN_CORPUS.jsonl" ] && [ -f "/home/adam/worxpace/gladius/raven/corpus/BOOK_OF_ARC.md" ]; then
        echo "Scribe has delivered the assets. Terminating monitor."
        exit 0
    fi
    echo "Scribe still hunting... checking again in 30s."
    sleep 30
done
