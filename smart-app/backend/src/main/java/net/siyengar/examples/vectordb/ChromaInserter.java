package net.siyengar.examples.vectordb;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import static com.example.Chroma.embeddingModel;
import static com.example.Chroma.embeddingStore;

public class ChromaInserter {

    /**
     * Add text.
     */
    public static void addDocuments(String text) {
        TextSegment segment1 = TextSegment.from(text, new Metadata());
        Embedding embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);
    }

    /**
     * Add text with metadata.
     */
    public static void addDocuments(String text, Metadata metadata) {
        TextSegment segment1 = TextSegment.from(text, metadata);
        Embedding embedding1 = embeddingModel.embed(segment1).content();
        embeddingStore.add(embedding1, segment1);
    }
}