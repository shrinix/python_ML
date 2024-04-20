package net.siyengar;

import static java.time.Duration.ofSeconds;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import net.siyengar.agent.CustomerSupportAgent;
import net.siyengar.agent.EmployeeTools;

@EnableJpaRepositories("net.siyengar.repository")
@SpringBootApplication(scanBasePackages={"net.siyengar.controller", "net.siyengar.service", "net.siyengar.repository", "net.siyengar.exception",
  "net.siyengar.model", "net.siyengar.agent"})
public class SpringbootBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(SpringbootBackendApplication.class, args);
	}

	//  @Bean
    // CustomerSupportAgent customerSupportAgent(ChatLanguageModel chatLanguageModel,
    //                                           EmployeeTools employeeTools) {

	// 	System.out.println("Creating CustomerSupportAgent bean");
    //    ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(20);


	// 	// ChatLanguageModel model = OllamaChatModel.builder()
    //     //         .baseUrl(BASE_URL)
    //     //         .modelName(MODEL)
    //     //         .timeout(timeout)
    //     //         .build();

	// @Bean
    // CustomerSupportAgent customerSupportAgent(EmployeeTools employeeTools) {
    //     ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(20);
	// 	ChatLanguageModel chatLanguageModel = OpenAiChatModel.builder()
	// 				.apiKey("demo")
	// 				.timeout(ofSeconds(60))
	// 				.temperature(0.0)
	// 				.build();

    //     return AiServices.builder(CustomerSupportAgent.class)
    //         .chatLanguageModel(chatLanguageModel)
    //         .chatMemory(chatMemory)
    //         .tools(employeeTools)
    //         .build();
    // }

}
