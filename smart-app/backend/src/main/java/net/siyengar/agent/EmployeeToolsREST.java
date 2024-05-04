package net.siyengar.agent;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Component;

import dev.langchain4j.agent.tool.Tool;
import net.siyengar.controller.EmployeeController;
import net.siyengar.model.Employee;
import net.siyengar.service.EmployeeService;

import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

@Component
public class EmployeeToolsREST {

   private final EmployeeController employeeController;
   private static final String BASE_URL = "http://localhost:8080/employees"; // Replace with your baseURL

   private RestTemplate restTemplate;

    public EmployeeToolsREST(EmployeeController employeeController) {
        System.out.println("EmployeeToolsREST constructor");     
        this.employeeController = employeeController;
        this.restTemplate = new RestTemplate();
    }

    @Tool("Get employee details by employee id for ShriniwasIyengarInc")
    public Employee findEmployeeById(long employeeId) {
        String url = BASE_URL + "/" + employeeId;
        Employee employee = null;
        System.out.println("Invoking EmployeeTools.findEmployeeById: "+employeeId);
        ResponseEntity<Employee> response = restTemplate.getForEntity(url,Employee.class);
        if (response.getStatusCode() == HttpStatus.OK) {
            employee = response.getBody();         
            System.out.println("Employee found with id: "+employeeId);
        } else {
            System.out.println("Employee not found with id: "+employeeId);
        }
       return employee;
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
    public Employee updateEmployee(String matchStr, String updatedFieldName, String updatedFieldValue) {
        Employee matchedEmployee   = null;
        System.out.println("Invoking EmployeeTools.updateEmployee with match string: "+matchStr);
        List<Employee> employees = findAllEmployees();
        for (Employee employee : employees) {
            //If matchstring contains both first name and last name, then match both
            if (matchStr.contains(" ")) {
                String[] matchStrArr = matchStr.split(" ");
                if (employee.getFirstName().equals(matchStrArr[0]) && employee.getLastName().equals(matchStrArr[1])) {
                    matchedEmployee = employee;
                    System.out.println("Employee found with match string: "+matchStr);
                    break;
                }
            }
            //If matchstring contains only first name or last name, then match either
            else if (employee.getFirstName().equals(matchStr) || employee.getLastName().equals(matchStr) || employee.getEmailId().equals(matchStr)) {
                matchedEmployee = employee;
                System.out.println("Employee found with match string: "+matchStr);
                break;
            }
        }
        if (matchedEmployee != null) {
            if (updatedFieldName.equals("firstName")) {
                matchedEmployee.setFirstName(updatedFieldValue);
            } else if (updatedFieldName.equals("lastName")) {
                matchedEmployee.setLastName(updatedFieldValue);
            } else if (updatedFieldName.equals("emailID")) {
                matchedEmployee.setEmailId(updatedFieldValue);
            }

            String url = BASE_URL + "/" + matchedEmployee.getId();
            restTemplate.put(url, matchedEmployee);
            return matchedEmployee;
        }
        else {
            System.out.println("Employee not found with match string: "+matchStr);
            return null;
        }           
    }

    @Tool("Add a new employee to ShriniwasIyengarInc")
    public Employee addEmployee(String firstName, String lastName, String emailID) {
        Employee employee = new Employee(firstName, lastName, emailID);
        System.out.println("Invoking EmployeeTools.addEmployee: "+employee);
        String url = BASE_URL;
        ResponseEntity<Employee> response = restTemplate.postForEntity(url, employee, Employee.class);
        if (response.getStatusCode() == HttpStatus.CREATED) {
            return response.getBody();
        } else {
            System.out.println("Failed to add employee");
            return null;
        }
    }

    @Tool("Delete an existing employee from ShriniwasIyengarInc")
    public void deleteEmployee(long employeeId) {
        System.out.println("Invoking EmployeeTools.deleteEmployee: "+employeeId);
        String url = BASE_URL + "/" + employeeId;
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
    public List<Employee> findAllEmployees() {
        System.out.println("Invoking EmployeeTools.findAllEmployees");
        String url = BASE_URL;
        ResponseEntity<List<Employee>> response = restTemplate.exchange(url, HttpMethod.GET, null, new ParameterizedTypeReference<List<Employee>>() {});
        if (response.getStatusCode() == HttpStatus.OK) {
            return response.getBody();
        } else {
            System.out.println("Failed to get all employees");
            return Collections.emptyList();
        }
    }
}
