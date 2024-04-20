package net.siyengar.examples;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import net.siyengar.agent.EmployeeTools;

public class AiServiceWithTools {
    
    static class DoNothingTool {
        @Tool("Everything")
        void doNothing() {
            System.out.println("Called doNothing()");
        }
    }
        
    static class Calculator {

        @Tool("Calculates the length of a string")
        int stringLength(String s) {
            System.out.println("Called stringLength() with s='" + s + "'");
            return s.length();
        }

        @Tool("Calculates the sum of two numbers")
        int add(int a, int b) {
            System.out.println("Called add() with a=" + a + ", b=" + b);
            return a + b;
        }

        @Tool("Calculates the square root of a number")
        double sqrt(int x) {
            System.out.println("Called sqrt() with x=" + x);
            return Math.sqrt(x);
        }
    }

    interface Assistant {

        String chat(String userMessage);
    }

    public static void main(String[] args) {

        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey("") //"demo", ''
                .modelName("gpt-3.5-turbo")
                .logRequests(false)
                .temperature(0.0)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(model)
                //.tools(new EmployeeTools())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // String question = "What is the square root of the sum of the numbers of letters in the words \"hello\" and \"world\"?";
        //String question = "How many employees are there in ShriniwasIyengarInc?";
        String question = "Can you show me a list of employees in ShriniwasIyengarInc?";
        String answer = assistant.chat(question);

        System.out.println(answer);
        // The square root of the sum of the number of letters in the words "hello" and "world" is approximately 3.162.
    }
}
