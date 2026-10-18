set dotenv-load := true

feast-apply:
	cd src/feature_repo && feast apply

start-up: feast-apply
	set -euo pipefail && trap 'kill 0' EXIT

default:
	@just --list


