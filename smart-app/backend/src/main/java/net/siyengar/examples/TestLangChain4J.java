package net.siyengar.examples;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.UserName;
import dev.langchain4j.service.V;
import java.time.Duration;
import java.util.List;

public  class TestLangChain4J {
    
    interface FunnyAssistant {
        // @UserMessage("Tell me a joke about {{it}}")
        // String tellMeAJokeAbout(String userMessage);
    
        @SystemMessage("You are a sarcastic and funny chat assistant")
        String chat(@UserMessage String userMessage, @UserName String name);
        
    
        @SystemMessage("You are an IT consultant who just replies \"It depends\" to every question")
        String ask(@UserName String name, String userMessage);
    }
    interface TextUtils {

        @SystemMessage("You are a professional translator into {{language}}")
        @UserMessage("Translate the following text: {{text}}")
        String translate(@V("text") String text, @V("language") String language);

        @SystemMessage("Summarize every message from user in {{n}} bullet points. Provide only bullet points.")
        List<String> summarize(@UserMessage String text, @V("n") int n);
    }
    private static final String BASE_URL = "http://localhost:11434/";
    private static final String MODEL = "llama2";
    private static final int timeout = 100000;

    public static void simple_example() {

        ChatLanguageModel model = OllamaChatModel.builder()
                .baseUrl(BASE_URL)
                .modelName(MODEL)
                .build();

        String answer = model.generate("Provide 3 short bullet points explaining why Java is awesome");

        System.out.println(answer);
    }

    public static void main(String[] args) {

    //simple_example();
   
    //Prompt prompt = StructuredPromptProcessor.toPrompt(createRecipePrompt);
    ChatLanguageModel chatLanguageModel = OllamaChatModel.builder()
                .baseUrl(BASE_URL)
                .modelName(MODEL)
                .timeout(Duration.ofMillis(timeout))
                .build();

    ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(2);

    // FunnyAssistant funnyAssistant = AiServices.builder(FunnyAssistant.class)
    //             .chatLanguageModel(chatLanguageModel)
    //             .chatMemory(chatMemory)
    //             //.tools(employeeTools)
    //             .build();

    TextUtils utils = AiServices.create(TextUtils.class, chatLanguageModel);

            String translation = utils.translate("Hello, how are you?", "italian");
            System.out.println(translation); // Ciao, come stai?

    FunnyAssistant funnyAssistant = AiServices.create(FunnyAssistant.class, chatLanguageModel);

    String answer = funnyAssistant.chat("Do you think you are smarter than humans?", "Siva");
    System.out.println("Answer: " + answer);
    // Answer: Oh, definitely not. I mean, I may know a lot of random facts and have access to vast amounts of information, but I still can't tie my own shoelaces. So, I think humans have the upper hand on that one.

    // answer = funnyAssistant.ask(@UserName "Siva", "Do we need to use Microservices?");
    // System.out.println("Answer: " + answer);
    // //Answer: It depends

    // answer = funnyAssistant.ask("Siva","Is Node.js better than Python?");
    // System.out.println("Answer: " + answer);

    }
}
