package net.siyengar.agent;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Component;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import com.fasterxml.jackson.databind.ObjectMapper;

@Component
public class EmployeeToolsREST {

   private static final String BASE_URL = "http://localhost:8080";

   private RestTemplate restTemplate;

    public EmployeeToolsREST() {
        System.out.println("EmployeeToolsREST constructor");     
        this.restTemplate = new RestTemplate();
    }

    @Tool("Add a new employee to ShriniwasIyengarInc")
    public Map<String, Object> createEmployee(String firstName, String lastName, String emailId) {
        System.out.println("Invoking EmployeeTools.createEmployee: "+firstName+" "+lastName+" "+emailId);
        try {
            // Create the employee map
            Map<String, Object> employee = new HashMap<>();
            employee.put("firstName", firstName);
            employee.put("lastName", lastName);
            employee.put("emailId", emailId);

            // Serialize the employee map into a JSON string
            ObjectMapper objectMapper = new ObjectMapper();
            String json = objectMapper.writeValueAsString(employee);
            System.out.println("Employee data being sent to the REST endpoint: " + json);

            Map<String, Object> result = restTemplate.postForObject(BASE_URL + "/employees", employee, Map.class);
            System.out.println("Employee created successfully");
            return result;
        } catch (Exception e) {
            System.out.println("Error occurred while creating employee: " + e.getMessage());
            return null;
        }
    }

    @Tool("Get employee details by employee id for ShriniwasIyengarInc")
    public Map<String, Object> findEmployeeById(long employeeId) {
        System.out.println("Invoking findEmployeeById for ID: " + employeeId);
        try {
            Map<String, Object> result = restTemplate.getForObject(BASE_URL + "/employees/" + employeeId, Map.class);
            return result;
        } catch (Exception e) {
            System.out.println("Error occurred while fetching employee details: " + e.getMessage());
            return null;
        }
    }

    // @Tool("Get employee details by employee firstName for ShriniwasIyengarInc")
    // public Employee findEmployeeByFirstName(String firstName, String lastName) {
    //     System.out.println("Invoking EmployeeTools.findEmployeeByFirstName: "+firstName);
    //     List<Employee> employees = employeeService.getAllEmployees();
    //     for (Employee employee : employees) {
    //         if (employee.getFirstName().equals(firstName) && employee.getLastName().equals(lastName)) {
    //             return employee;
    //         }
    //     }
    //     System.out.println("Employee not found with name: "+firstName+""+lastName);
    //     return null;
    // }

    // @Tool("Get employee details by employee email ID for ShriniwasIyengarInc")
    // public Employee findEmployeeByEmailID(String emailID) {
    //     System.out.println("Invoking EmployeeTools.findEmployeeByLastName: "+emailID);
    //     List<Employee> employees = employeeService.findAllEmployees();
    //     for (Employee employee : employees) {
    //         if (employee.getLastName().equals(emailID)) {
    //             return employee;
    //         }
    //     }
    //     System.out.println("Employee not found with emailID: "+emailID);
    //     return null;
    // }

    @Tool("Update or modify an existing employee of ShriniwasIyengarInc using first name or last name or email ID")
    public Map<String, Object> updateEmployee(String matchStr, String updatedFieldName, String updatedFieldValue) {
        System.out.println("Invoking EmployeeTools.updateEmployee with match string: " + matchStr);
        List<Map<String, Object>> employees = findAllEmployees();
        for (Map<String, Object> employee : employees) {
            // If matchstring contains both first name and last name, then match both
            if (matchStr.contains(" ")) {
                String[] matchStrArr = matchStr.split(" ");
                if (employee.get("firstName").equals(matchStrArr[0]) && employee.get("lastName").equals(matchStrArr[1])) {
                    employee.put(updatedFieldName, updatedFieldValue);
                    String url = BASE_URL + "/employees/" + employee.get("id");
                    restTemplate.put(url, employee);
                    System.out.println("Employee updated successfully");
                    return employee;
                }
            }
            // If matchstring contains only first name or last name, then match either
            else if (employee.get("firstName").equals(matchStr) || employee.get("lastName").equals(matchStr) || employee.get("emailID").equals(matchStr)) {
                employee.put(updatedFieldName, updatedFieldValue);
                String url = BASE_URL + "/employees/" + employee.get("id");
                restTemplate.put(url, employee);
                System.out.println("Employee updated successfully");
                return employee;
            }
        }
        System.out.println("Employee not found with match string: " + matchStr);
        return null;
    }
    
    @Tool("Delete an existing employee from ShriniwasIyengarInc")
    public void deleteEmployee(long employeeId) {
        System.out.println("Invoking EmployeeTools.deleteEmployee: "+employeeId);
        String url = BASE_URL + "/employees/" + employeeId;
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.DELETE, null, Map.class);
        if (response.getStatusCode() == HttpStatus.OK) {
            System.out.println("Employee deleted successfully");
        } else {
            System.out.println("Failed to delete employee");
        }
    }

    // @Tool("Finds an existing employee by email")
    // public CustomerRecord findCustomerByEmail(String email) {
    //     return employeeService.findCustomertByEmail(email);
    // }

    @Tool("Gets a list of all employees in ShriniwasIyengarInc")
    public List<Map<String,Object>> findAllEmployees() {

        // Get all employees
        System.out.println("Invoking EmployeeTools.findAllEmployees");
        try {
            ResponseEntity<Map[]> response = restTemplate.getForEntity(BASE_URL + "/employees", Map[].class);
            List<Map<String,Object>> employees = Arrays.asList(response.getBody());
            System.out.println("Obtained "+employees.size()+" employees");
            return employees;
        } catch (Exception e) {
            System.out.println("Error occurred while fetching employees: " + e.getMessage());
            return null;
        }
    }

    //Add a main method with code to test the EmployeeToolsREST class
    public static void main(String[] args) {
        EmployeeToolsREST employeeToolsREST = new EmployeeToolsREST();

        List<Map<String,Object>> employees = employeeToolsREST.findAllEmployees();
        System.out.println("Employees: "+employees);

        //Test Add Employee using some test data
        Map<String,Object> createdEmployee = employeeToolsREST.createEmployee("John","Doe","jdoe@xyz.com");
        System.out.println("Created Employee: "+createdEmployee);
        Map<String,Object> updatedEmployee = employeeToolsREST.updateEmployee("John Doe","emailId","jdoe@abc.com");
        System.out.println("Updated Employee: "+updatedEmployee);

        //Test Find Employee by ID
        Map<String,Object> employee = employeeToolsREST.findEmployeeById(1);
        System.out.println("Employee: "+employee);

        employees = employeeToolsREST.findAllEmployees();
        System.out.println("Employees: "+employees);

        //pick the first employee from the list
        Map<String,Object> firstEmployee = employees.get(0);
        int employeeId = (Integer) firstEmployee.get("id"); // Change the cast to Integer
        //Test Delete Employee
        employeeToolsREST.deleteEmployee(employeeId);

        employees = employeeToolsREST.findAllEmployees();
        System.out.println("Employees: "+employees);

    }
}