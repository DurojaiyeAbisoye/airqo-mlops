set dotenv-load := true

default:
	just start-up

format:
	autoflake --in-place --remove-all-unused-imports --recursive .
	isort .
	black .

start-up:
	#!/usr/bin/env bash
	set -euo pipefail

	state_dir=".air_quality_services"
	mkdir -p "$state_dir"

	start_service() {
		name="$1"
		command_pattern="$2"
		shift 2
		pid_file="$state_dir/$name.pid"

		if [ -f "$pid_file" ]; then
			pid="$(cat "$pid_file")"
			if kill -0 "$pid" 2>/dev/null; then
				process_command="$(ps -p "$pid" -o args=)"
				if [[ "$process_command" == *"$command_pattern"* ]]; then
					echo "$name is already running (PID $pid)"
					return
				fi
			fi
			rm -f "$pid_file"
		fi

		nohup "$@" >"$state_dir/$name.log" 2>&1 </dev/null &
		echo "$!" >"$pid_file"
		echo "Started $name (PID $!)"
	}

	start_service mlflow "mlflow ui" mlflow ui
	(cd src/feature_repo && feast apply)

	start_service evidently "evidently ui" evidently ui
	start_service feast "feast ui" bash -c 'cd src/feature_repo && exec feast ui'

shutdown:
	#!/usr/bin/env bash
	set -euo pipefail

	state_dir=".air_quality_services"
	for name in mlflow evidently feast; do
		pid_file="$state_dir/$name.pid"
		if [ ! -f "$pid_file" ]; then
			echo "$name is not running (no PID file)"
			continue
		fi

		pid="$(cat "$pid_file")"
		if ! kill -0 "$pid" 2>/dev/null; then
			echo "$name is not running (stale PID file)"
			rm -f "$pid_file"
			continue
		fi

		process_command="$(ps -p "$pid" -o args=)"
		if [[ "$process_command" != *"$name ui"* ]]; then
			echo "Refusing to stop PID $pid: it does not appear to be $name ui" >&2
			exit 1
		fi

		kill "$pid"
		rm -f "$pid_file"
		echo "Stopped $name (PID $pid)"
	done


feast-apply: format
	cd src/feature_repo && feast apply

fetch-daily-sensor-median: format
	python src/data_pipeline/fetch_daily_sensor_median.py

fetch-lon-lat: format
	python src/data_pipeline/fetch_lon_lat.py

fetch-historical-weather: format
	python src/data_pipeline/fetch_historical_weather.py

fetch-forecast-weather: format
	python src/data_pipeline/fetch_forecast_weather_data.py

merge-data: format fetch-historical-weather fetch-forecast-weather fetch-daily-sensor-median
	python src/data_pipeline/merge_sensor_data.py
	python src/data_pipeline/merge_weather_data.py


baseline: merge-data
	python src/training_pipeline/baseline.py

tune:
	python src/training_pipeline/tune.py

promote-model: tune
	python src/training_pipeline/promote_model.py

inference:  promote-model
	python src/batch_inference_pipeline/inference.py

monitor: inference
	python src/batch_inference_pipeline/monitoring.py && python src/batch_inference_pipeline/report.py
