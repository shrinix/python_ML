package net.siyengar.controller;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
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
import net.siyengar.agent.EmployeeTools;
import net.siyengar.model.Employee;
import net.siyengar.model.ChatResponse;
import net.siyengar.service.EmployeeService;
import static java.time.Duration.ofSeconds;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
//@RequestMapping("/api/v1/")
public class EmployeeController {

	private static final String BASE_URL = "http://localhost:11434/";
    private static final String MODEL = "llama2";
    private static final int timeout = 100000;
	private CustomerSupportAgent assistant;

	@Autowired
	private final EmployeeService employeeService;
	
	//enum for the model type
	private enum ModelType {
		OLLAMA,
		OPENAI
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
		AiServices.builder(CustomerSupportAgent.class)
		.chatLanguageModel(OpenAiChatModel.builder()
				.apiKey(System.getenv("API_KEY")) // use environment variable for apiKey
				.timeout(ofSeconds(60))
				.temperature(0.0)
				.build())
			.tools(new EmployeeTools(employeeService))
			.chatMemory(MessageWindowChatMemory.withMaxMessages(10))
			.build();
		System.out.println("Finished creating CustomerSupportAgent bean");
		return assistant;
	}


    public EmployeeController(EmployeeService employeeService) {
		this.employeeService = employeeService;
		assert employeeService != null;
		
		ChatLanguageModel chatLanguageModel = createChatLanguageModel(ModelType.OPENAI);
		this.assistant = createAssistant(chatLanguageModel);
			
		System.out.println("Finished creating CustomerSupportAgent bean");
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
	// get all employees
	@GetMapping("/employees")
	public List<Employee> getAllEmployees(){
		return employeeService.findAllEmployees();
	}		
	
	// create employee rest api
	@PostMapping("/employees")
	public Employee createEmployee(@RequestBody Employee employee) {
		return employeeService.saveEmployee(employee);
	}
	
	// get employee by id rest api
	@GetMapping("/employees/{id}")
	public ResponseEntity<Employee> getEmployeeById(@PathVariable Long id) {
		Employee employee = employeeService.findEmployeeById(id);
		return ResponseEntity.ok(employee);
	}
	
	// // get employee by email id
	// @GetMapping("/employees/{email_id}")
	// public ResponseEntity<Employee> getEmployeeByEmailId(@PathVariable String emailId) {
	// 	Employee employee = employeeService.findEmployeeByEmailId(emailId);
	// 	return ResponseEntity.ok(employee);
	// }

	// update employee via rest api
	@PutMapping("/employees/{id}")
	public ResponseEntity<Employee> updateEmployee(@PathVariable Long id, @RequestBody Employee employeeDetails){

		Employee updatedEmployee = employeeService.updateEmployee(id, employeeDetails);
		return ResponseEntity.ok(updatedEmployee);
	}
	
	// delete employee rest api
	@DeleteMapping("/employees/{id}")
	public ResponseEntity<Map<String, Boolean>> deleteEmployee(@PathVariable Long id){
		
		employeeService.deleteEmployee(id);

		Map<String, Boolean> response = new HashMap<>();
		response.put("deleted", Boolean.TRUE);
		return ResponseEntity.ok(response);
	}
	
	
}
