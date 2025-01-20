#!/usr/bin/env bash

action() {
    source "{{analysis_path}}/setup.sh" "$@"
}
action "$@"