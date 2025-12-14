#! /bin/bash

docker compose down --volumes --remove-orphans && ./clear_data.sh && ./build_and_run.sh
