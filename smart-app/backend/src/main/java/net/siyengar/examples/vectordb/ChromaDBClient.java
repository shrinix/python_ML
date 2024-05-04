package net.siyengar.examples.vectordb;

import com.google.gson.internal.LinkedTreeMap;
import tech.amikos.chromadb.Client;
import tech.amikos.chromadb.Collection;
import tech.amikos.chromadb.EmbeddingFunction;
import tech.amikos.chromadb.OpenAIEmbeddingFunction;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
/**
 * Hello Chroma!
 *
 */
public class ChromaDBClient 
{
    public static void main(String[] args) {
        try {
            Properties properties = new Properties();
            try (InputStream inputStream = ChromaDBClient.class.getClassLoader().getResourceAsStream("application.properties")) {
                properties.load(inputStream);
            } catch (IOException e) {
                e.printStackTrace();
            }
            String chromaUrl = properties.getProperty("CHROMA_URL");
            Client client = new Client(chromaUrl);
            String apiKey = System.getenv("OPENAI_API_KEY");

            
            SentenceTransformerEmbeddings embeddings = new SentenceTransformerEmbeddings();

            EmbeddingFunction ef = new OpenAIEmbeddingFunction(apiKey);
            Collection collection = client.createCollection("test-collection", null, true, ef);
            List<Map<String, String>> metadata = new ArrayList<>();
            metadata.add(new HashMap<String, String>() {{
                put("type", "scientist");
            }});
            metadata.add(new HashMap<String, String>() {{
                put("type", "spy");
            }});
            collection.add(null, metadata, Arrays.asList("Hello, my name is John. I am a Data Scientist.", "Hello, my name is Bond. I am a Spy."), Arrays.asList("1", "2"));
            Collection.QueryResponse qr = collection.query(Arrays.asList("Who is the spy"), 10, null, null, null);
            System.out.println(qr);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e);
        }
    }
}