#!/bin/bash -f

# --- show usage ---
function usage {
    cat <<EOF
$(basename ${0}) is a test tool for launch streaming server

Usage:
    $(basename ${0}) [command] [<options>]

Commands:
    --task, -t        objective('through', 'object_detection')
Options:
    --version, -v     print version
    --help, -h        print this
EOF
}

# --- show version ---
function version {
    echo "$(basename ${0}) version 0.0.1 "
}

# --- argument processing ---
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

TASK='through'
while [ $# -gt 0 ];
do
    case ${1} in

        --task|-t)
            TASK=${2}
            shift
        ;;
        
        --version|-v)
            version
            exit 1
        ;;
        
        --help|-h)
            usage
            exit 1
        ;;
        
        *)
            echo "[ERROR] Invalid option '${1}'"
            usage
            exit 1
        ;;
    esac
    shift
done

# --- download tflite models ---
CLASS_LABEL=""
TFLITE_FILE=""
if [ ${TASK} = "object_detection" ]; then
    TFLITE_DIR="./models/efficientdet_lite0"
    mkdir -p ${TFLITE_DIR}
    TFLITE_FILE="${TFLITE_DIR}/efficientdet_lite0.tflite"
    CLASS_LABEL="${TFLITE_DIR}/labelmap.txt"
    if [ ! -f ${TFLITE_FILE} ]; then
        curl \
            -L 'https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/metadata/1?lite-format=tflite' \
	    -o ${TFLITE_FILE}
	unzip ${TFLITE_FILE} -d ${TFLITE_DIR}
    fi
fi

# --- launch server ---
STREAMING_TASK="${TASK}" \
TFLITE_FILE="${TFLITE_FILE}" \
CLASS_LABEL="${CLASS_LABEL}" \
mjpg_streamer -i "/usr/local/lib/mjpg-streamer/input_opencv.so --filter /usr/local/lib/mjpg-streamer/cvfilter_py.so --fargs ./main.py" -o "/usr/local/lib/mjpg-streamer/output_http.so -w ./www"


