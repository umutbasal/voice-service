package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	pb "voice-assistant/client/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	address   = "localhost:50051"
	uzbekText = "Салом,  хуш келибсиз!  Сиз Аҳмет Ин витро уруғлантириш ва репродуктив саломатлик марказига улангансиз,  мен Алиман.  Сизга қандай ёрдам беришим мумкин? Клиникамизда проф.  Др.  Аҳмет Каража ва жамоамиз билан энг замонавий даволаш усулларини таклиф этамиз.  Агар сиз учрашувга ёзилиш,  жараёнлар ҳақида билиш ёки маслаҳат хизматига мурожаат қилишни истасангиз,  биз сизга энг яхши тарзда ёрдам беришдан хурсанд бўламиз.  Мен сизни тинглаш учун келдим,  саволларингизга жавоб бера оламан."
)

func main() {
	// Set up a connection to the server
	conn, err := grpc.Dial(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()

	// Create clients
	ttsClient := pb.NewTextToSpeechServiceClient(conn)
	sttClient := pb.NewSpeechToTextServiceClient(conn)

	//Example 1: Simple Text-to-Speech
	fmt.Println("\n=== Simple Text-to-Speech ===")
	simpleTextToSpeech(ttsClient)

	// Example 2: Streaming Text-to-Speech
	fmt.Println("\n=== Streaming Text-to-Speech ===")
	streamingTextToSpeech(ttsClient)

	// Example 3: Stream Text Request to Speech
	fmt.Println("\n=== Stream Text Request to Speech ===")
	streamTextRequestToSpeech(ttsClient)

	// Example 4: Bidirectional Text-to-Speech Streaming
	fmt.Println("\n=== Bidirectional Text-to-Speech Streaming ===")
	bidirectionalTextToSpeech(ttsClient)

	// Example 5: Simple Speech-to-Text
	fmt.Println("\n=== Simple Speech-to-Text ===")
	simpleSpeechToText(sttClient)

	// Example 6: Streaming Speech-to-Text
	fmt.Println("\n=== Streaming Speech-to-Text ===")
	streamingSpeechToText(sttClient)

	// Example 7: Stream Speech Request to Text
	fmt.Println("\n=== Stream Speech Request to Text ===")
	streamSpeechRequestToText(sttClient)

	// Example 8: Bidirectional Speech-to-Text Streaming
	fmt.Println("\n=== Bidirectional Speech-to-Text Streaming ===")
	bidirectionalSpeechToText(sttClient)
}

func simpleTextToSpeech(client pb.TextToSpeechServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	text := "Салом, хуш келибсиз!"
	resp, err := client.UnaryConvertTextToSpeech(ctx, &pb.TextToSpeechRequest{
		Text:     text,
		Language: "uz",
		Voice:    "default",
	})
	if err != nil {
		log.Fatalf("could not convert text to speech: %v", err)
	}

	// print text
	fmt.Printf("Processed Text: %s\n", text)
	// Save the audio to a file
	err = os.WriteFile("output.wav", resp.AudioData, 0644)
	if err != nil {
		log.Fatalf("could not save audio file: %v", err)
	}
	fmt.Println("Audio saved to output.wav")
}

func streamingTextToSpeech(client pb.TextToSpeechServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	stream, err := client.ServerStreamTextToSpeech(ctx, &pb.TextToSpeechRequest{
		Text:     uzbekText,
		Language: "uz",
		Voice:    "default",
	})
	if err != nil {
		log.Fatalf("could not stream text to speech: %v", err)
	}

	// print text
	fmt.Printf("Processed Text: %s\n", uzbekText)

	chunkCount := 0
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("error receiving chunk: %v", err)
		}
		chunkCount++
		fmt.Printf("Received audio chunk %d of size: %d bytes\n", chunkCount, len(chunk.AudioData))

		// Save the audio to a file
		err = os.WriteFile(fmt.Sprintf("output_streamed_%d.wav", chunkCount), chunk.AudioData, 0644)
		if err != nil {
			log.Fatalf("could not save audio file: %v", err)
		}

		fmt.Printf("Saved audio chunk %d to output_streamed_%d.wav\n", chunkCount, chunkCount)
	}
}

func streamTextRequestToSpeech(client pb.TextToSpeechServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	stream, err := client.ClientStreamTextToSpeech(ctx)
	if err != nil {
		log.Fatalf("could not create stream: %v", err)
	}

	// Send multiple text chunks
	texts := []string{
		"Салом, хуш келибсиз!",
		"Сиз Аҳмет Ин витро уруғлантириш ва репродуктив саломатлик марказига улангансиз.",
		"Мен сизни тинглаш учун келдим, саволларингизга жавоб бера оламан.",
	}

	for _, text := range texts {
		if err := stream.Send(&pb.TextChunk{Text: text}); err != nil {
			log.Fatalf("could not send text chunk: %v", err)
		}
		fmt.Printf("Sent text chunk: %s\n", text)
	}

	resp, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("error receiving response: %v", err)
	}

	// Save the combined audio to a file
	err = os.WriteFile("output_combined.wav", resp.AudioData, 0644)
	if err != nil {
		log.Fatalf("could not save audio file: %v", err)
	}
	fmt.Println("Combined audio saved to output_combined.wav")
}

func bidirectionalTextToSpeech(client pb.TextToSpeechServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	stream, err := client.BidirectionalStreamTextToSpeech(ctx)
	if err != nil {
		log.Fatalf("could not create stream: %v", err)
	}

	// Create a channel to receive responses
	waitc := make(chan struct{})

	// Start goroutine to receive audio chunks
	go func() {
		chunkCount := 0
		for {
			chunk, err := stream.Recv()
			if err == io.EOF {
				close(waitc)
				return
			}
			if err != nil {
				log.Fatalf("error receiving chunk: %v", err)
			}
			chunkCount++
			fmt.Printf("Received audio chunk %d of size: %d bytes\n", chunkCount, len(chunk.AudioData))

			// Save the audio to a file
			err = os.WriteFile(fmt.Sprintf("output_bidirectional_%d.wav", chunkCount), chunk.AudioData, 0644)
			if err != nil {
				log.Fatalf("could not save audio file: %v", err)
			}
			fmt.Printf("Saved audio chunk %d to output_bidirectional_%d.wav\n", chunkCount, chunkCount)
		}
	}()

	// Send text chunks
	texts := []string{
		"Клиникамизда проф. Др. Аҳмет Каража ва жамоамиз билан",
		"энг замонавий даволаш усулларини таклиф этамиз.",
		"биз сизга энг яхши тарзда ёрдам беришдан хурсанд бўламиз.",
	}

	for _, text := range texts {
		if err := stream.Send(&pb.TextChunk{Text: text}); err != nil {
			log.Fatalf("could not send text chunk: %v", err)
		}
		fmt.Printf("Sent text chunk: %s\n", text)
		time.Sleep(time.Second) // Add delay between sends
	}

	stream.CloseSend()
	<-waitc // Wait for receiving to complete
}

func simpleSpeechToText(client pb.SpeechToTextServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	// Read audio file
	audioData, err := os.ReadFile("input.wav")
	if err != nil {
		log.Fatalf("could not read audio file: %v", err)
	}

	resp, err := client.UnaryConvertSpeechToText(ctx, &pb.SpeechToTextRequest{
		AudioData: audioData,
	})
	if err != nil {
		log.Fatalf("could not convert speech to text: %v", err)
	}

	fmt.Printf("Transcription: %s\n", resp.TranscribedText)
}

func streamingSpeechToText(client pb.SpeechToTextServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	// Read audio file
	audioData, err := os.ReadFile("input.wav")
	if err != nil {
		log.Fatalf("could not read audio file: %v", err)
	}

	stream, err := client.ServerStreamSpeechToText(ctx, &pb.SpeechToTextRequest{
		AudioData: audioData,
	})
	if err != nil {
		log.Fatalf("could not stream speech to text: %v", err)
	}

	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("error receiving transcription: %v", err)
		}
		fmt.Printf("Partial transcription: %s\n", resp.PartialTranscription)
	}
}

func streamSpeechRequestToText(client pb.SpeechToTextServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	stream, err := client.ClientStreamSpeechToText(ctx)
	if err != nil {
		log.Fatalf("could not create stream: %v", err)
	}

	// Read and send multiple chunk files
	for i := 1; ; i++ {
		filename := fmt.Sprintf("chunk_%d.wav", i)
		audioData, err := os.ReadFile(filename)
		if err != nil {
			if os.IsNotExist(err) {
				break // No more chunks
			}
			log.Fatalf("could not read audio file %s: %v", filename, err)
		}

		// Send the entire WAV file as one chunk
		if err := stream.Send(&pb.AudioChunk{AudioData: audioData}); err != nil {
			log.Fatalf("could not send audio chunk: %v", err)
		}
		fmt.Printf("Sent chunk file %s, size: %d bytes\n", filename, len(audioData))
	}

	resp, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("error receiving response: %v", err)
	}

	fmt.Printf("Final transcription: %s\n", resp.TranscribedText)
}

func bidirectionalSpeechToText(client pb.SpeechToTextServiceClient) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	stream, err := client.BidirectionalStreamSpeechToText(ctx)
	if err != nil {
		log.Fatalf("could not create stream: %v", err)
	}

	waitc := make(chan struct{})

	// Start goroutine to receive transcriptions
	go func() {
		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				close(waitc)
				return
			}
			if err != nil {
				log.Fatalf("error receiving transcription: %v", err)
			}
			fmt.Printf("Received transcription: %s\n", resp.PartialTranscription)
		}
	}()

	// Read and send multiple chunk files
	for i := 1; ; i++ {
		filename := fmt.Sprintf("chunk_%d.wav", i)
		audioData, err := os.ReadFile(filename)
		if err != nil {
			if os.IsNotExist(err) {
				break // No more chunks
			}
			log.Fatalf("could not read audio file %s: %v", filename, err)
		}

		// Send the entire WAV file as one chunk
		if err := stream.Send(&pb.AudioChunk{AudioData: audioData}); err != nil {
			log.Fatalf("could not send audio chunk: %v", err)
		}
		fmt.Printf("Sent chunk file %s, size: %d bytes\n", filename, len(audioData))
		time.Sleep(time.Millisecond * 100) // Add delay between sends
	}

	stream.CloseSend()
	<-waitc // Wait for receiving to complete
}
