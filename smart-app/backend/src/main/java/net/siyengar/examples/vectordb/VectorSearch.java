package net.siyengar.examples.vectordb;

import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel; 
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
//import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

import static dev.langchain4j.internal.Utils.randomUUID;

public class VectorSearch {
    
    public VectorSearch() {
        
        // VectorStore vectorStore = null;
        // EmbeddingClient embeddingClient = null;
        // TextReader textReader = null;
        
        // // Initialize embedding client
        // embeddingClient = new AllMiniLmL6V2EmbeddingModel();
        // // Initialize vector store
        // vectorStore = new SimpleVectorStore(embeddingClient);

        // EmbeddingStore<TextSegment> embeddingStore = ChromaEmbeddingStore.builder()
        //         .baseUrl(chroma.getEndpoint())
        //         .collectionName(randomUUID())
        //         .build();

    }

    // public void storeAndRetrieveEmbeddings() {
    //     // Store embeddings
    //     List<Document> documents = 
    //             List.of(new Document("I like Spring Boot"),
    //                     new Document("I love Java programming language"));
    //     vectorStore.add(documents);
        
    //     // Retrieve embeddings
    //     SearchRequest query = SearchRequest.query("Spring Boot").withTopK(2);
    //     List<Document> similarDocuments = vectorStore.similaritySearch(query);
    //     String relevantData = similarDocuments.stream()
    //                         .map(Document::getContent)
    //                         .collect(Collectors.joining(System.lineSeparator()));
    // }

    // public void convertTextToEmbedding() {
    //     //Example 1: Convert text to embeddings
    //     List<Double> embeddings1 = embeddingClient.embed("I like Spring Boot");
        
    //     //Example 2: Convert document to embeddings
    //     List<Double> embeddings2 = embeddingClient.embed(new Document("I like Spring Boot"));
        
    //     //Example 3: Convert text to embeddings using options
    //     EmbeddingRequest embeddingRequest =
    //             new EmbeddingRequest(List.of("I like Spring Boot"),
    //                     OpenAiEmbeddingOptions.builder()
    //                             .withModel("text-davinci-003")
    //                             .build());
    //     EmbeddingResponse embeddingResponse = embeddingClient.call(embeddingRequest);
    //     List<Double> embeddings3 = embeddingResponse.getResult().getOutput();
    // }

    // public static void main(String[] args) {
        
    //     try {
    //         Properties properties = new Properties();
    //         try (InputStream inputStream = VectorSearch.class.getClassLoader().getResourceAsStream("application.properties")) {
    //             properties.load(inputStream);
    //         } catch (IOException e) {
    //             e.printStackTrace();
    //         }
    //         String chromaUrl = properties.getProperty("CHROMA_URL");
            
    //         EmbeddingStore<TextSegment> embeddingStore = ChromaEmbeddingStore.builder()
    //                 .baseUrl(chromaUrl)
    //                 .collectionName(randomUUID())
    //                 .build();

    //         EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

    //         TextSegment segment1 = TextSegment.from("I like football.");
    //         Embedding embedding1 = embeddingModel.embed(segment1).content();
    //         embeddingStore.add(embedding1, segment1);

    //         TextSegment segment2 = TextSegment.from("The weather is good today.");
    //         Embedding embedding2 = embeddingModel.embed(segment2).content();
    //         embeddingStore.add(embedding2, segment2);

    //         Embedding queryEmbedding = embeddingModel.embed("What is your favourite sport?").content();
    //         List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(queryEmbedding, 1);
    //         EmbeddingMatch<TextSegment> embeddingMatch = relevant.get(0);

    //         System.out.println(embeddingMatch.score()); // 0.8144288493114709
    //         System.out.println(embeddingMatch.embedded().text()); // I like football.
    //     } catch (Exception e) {
    //         e.printStackTrace();
    //         System.out.println(e);
    //     }
    // }
} 
