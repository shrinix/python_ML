package net.siyengar.controller;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import net.siyengar.agent.CustomerSupportAgent;
import net.siyengar.agent.EmployeeToolsREST;
import net.siyengar.model.ChatResponse;
import static java.time.Duration.ofSeconds;

@RestController
//@RequestMapping("/api/v1/")
public class ChatController {

	private static final String BASE_URL = "http://localhost:11434/";
    private static final String MODEL = "llama2";
    private static final int timeout = 100000;
	private CustomerSupportAgent assistant;

	@Value("${cors.origin}")
    private String corsOrigin;
	
	//enum for the model type
	private enum ModelType {
		OLLAMA,
		OPENAI
	}

	public ChatController() {

		ChatLanguageModel chatLanguageModel = createChatLanguageModel(ModelType.OPENAI);
		this.assistant = createAssistant(chatLanguageModel);
		assert this.assistant != null;
			
		System.out.println("Finished creating CustomerSupportAgent bean");
    }

	private ChatLanguageModel createChatLanguageModel(ModelType modelType) {

		System.out.println("Creating ChatLanguageModel: "+modelType);
		ChatLanguageModel chatLanguageModel = null;

		switch (modelType) {
			case OLLAMA:
				chatLanguageModel = OllamaChatModel.builder()
					.baseUrl(BASE_URL)
					.modelName(MODEL)
					.timeout(Duration.ofMillis(timeout))
					.build();
				break;
			case OPENAI:
				chatLanguageModel = OpenAiChatModel.builder()
					.apiKey("demo")
					.timeout(ofSeconds(60))
					.temperature(0.0)
					.build();
				break;
			default:
				break;
		}
		
		return chatLanguageModel;
	}
	
	private CustomerSupportAgent createAssistant(ChatLanguageModel chatLanguageModel) {

		EmployeeToolsREST employeeToolsRest = new EmployeeToolsREST();
		System.out.println("Finished creating CustomerSupportAgent bean");

		CustomerSupportAgent assistant = AiServices.builder(CustomerSupportAgent.class)
		        .chatLanguageModel(OpenAiChatModel.builder()
				.apiKey(System.getenv("API_KEY")) // use environment variable for apiKey
				.timeout(ofSeconds(60))
				.temperature(0.0)
				.build())
				.tools(employeeToolsRest)
				.chatMemory(MessageWindowChatMemory.withMaxMessages(10))
				.build();
		System.out.println("Finished creating chat language model");
		return assistant;
	}

	@GetMapping("/generate")
	public ResponseEntity<ChatResponse> generate(
		@RequestParam(value = "userMessage", defaultValue = "Why is the sky blue?")
			String userMessage) {
		System.out.println("User message: " + userMessage);
		String answer = assistant.chat(userMessage, "Shrini");
		System.out.println("Model response: " + answer);
		ChatResponse chatResponse = new ChatResponse();
		chatResponse.setMessage(answer);
		System.out.println("Chat response: " + chatResponse.toString());
		return ResponseEntity.status(HttpStatus.OK).body(chatResponse);
	}
	
}
