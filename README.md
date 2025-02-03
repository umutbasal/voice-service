# Voice Service

A Proof of Concept (PoC) implementation demonstrating various gRPC streaming patterns for voice services. This project showcases all four types of gRPC communication (Unary, Server Streaming, Client Streaming, and Bidirectional Streaming) for both Text-to-Speech (TTS) and Speech-to-Text (STT) operations. Built with a Go client and Python server implementation.

## Features

### **Text-to-Speech (TTS) and Speech-to-Text (STT) - All gRPC Patterns**

- Unary: Simple one-request-one-response TTS - STT
- Server Streaming: Single text or audio input, continuous partial transcriptions
- Client Streaming: Multiple text or audio chunks, single combined transcription
- Bidirectional Streaming: Continuous exchange of text and audio
