# Voice Service

A Proof of Concept (PoC) implementation demonstrating various gRPC streaming patterns for voice services. This project showcases all four types of gRPC communication (Unary, Server Streaming, Client Streaming, and Bidirectional Streaming) for both Text-to-Speech (TTS) and Speech-to-Text (STT) operations. Built with a Go client and Python server implementation.

## Features

### **Text-to-Speech (TTS)**

- Unary: Simple one-request-one-response TTS
- Server Streaming: Single text input, continuous partial transcriptions
- Client Streaming: Multiple text chunks, single combined transcription
- Bidirectional Streaming: Continuous exchange of text and audio

### **Speech-to-Text (STT)**

- Unary: Simple one-request-one-response STT
- Server Streaming: Single audio input, continuous partial transcriptions
- Client Streaming: Multiple audio chunks, single combined transcription
- Bidirectional Streaming: Continuous exchange of audio and text
