.PHONY: proto clean

# Variables
PROTO_FILE := voice.proto
PROTO_PATH := .
GO_OUT_DIR := client/proto
PY_OUT_DIR := .

# Check if protoc is installed
PROTOC := $(shell command -v protoc 2> /dev/null)
ifndef PROTOC
    $(error "protoc is not installed. Please install Protocol Buffers compiler")
endif

proto: proto-go proto-python

proto-go:
	@echo "Generating Go protobuf files..."
	@mkdir -p $(GO_OUT_DIR)
	protoc --proto_path=$(PROTO_PATH) \
		--go_out=$(GO_OUT_DIR) \
		--go_opt=paths=source_relative \
		--go-grpc_out=$(GO_OUT_DIR) \
		--go-grpc_opt=paths=source_relative \
		$(PROTO_FILE)
	@echo "Go protobuf files generated successfully"

proto-python:
	@echo "Generating Python protobuf files..."
	python -m grpc_tools.protoc \
		--proto_path=$(PROTO_PATH) \
		--python_out=$(PY_OUT_DIR) \
		--grpc_python_out=$(PY_OUT_DIR) \
		$(PROTO_FILE)
	@echo "Python protobuf files generated successfully"

clean:
	@echo "Cleaning generated protobuf files..."
	rm -f $(GO_OUT_DIR)/*.pb.go
	rm -f $(PY_OUT_DIR)/*_pb2*.py
	@echo "Cleaned generated files" 