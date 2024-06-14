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

import net.siyengar.model.Employee;
import net.siyengar.model.ChatResponse;
import net.siyengar.service.EmployeeService;
import static java.time.Duration.ofSeconds;

//@CrossOrigin(origins = "${cors.origin}")
@RestController
//@RequestMapping("/api/v1/")
public class EmployeeController {

	private static final String BASE_URL = "http://localhost:11434/";
    private static final int timeout = 100000;

	@Value("${cors.origin}")
    private String corsOrigin;
	
	@Autowired
	private final EmployeeService employeeService;
	

    public EmployeeController(EmployeeService employeeService) {

		this.employeeService = employeeService;
		assert employeeService != null;
    }

	// get all employees
	@GetMapping("/employees")
	public List<Employee> getAllEmployees(){
		return employeeService.findAllEmployees();
	}		
	
	// create employee rest api
	@PostMapping("/employees")
	public Employee createEmployee(@RequestBody Employee employee) {
		System.out.println("Employee to be created: "+employee.toString());
		Employee createdEmployee = employeeService.saveEmployee(employee);
		System.out.println("Employee created successfully: "+createdEmployee.toString());
		return createdEmployee;
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
