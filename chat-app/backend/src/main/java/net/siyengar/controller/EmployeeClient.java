package net.siyengar.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EmployeeClient {

    private static final String BASE_URL = "http://localhost:8080"; // replace with your server's URL and port

    public static void main(String[] args) {
        RestTemplate restTemplate = new RestTemplate();

        // Get all employees
        ResponseEntity<Map[]> response = restTemplate.getForEntity(BASE_URL + "/employees", Map[].class);
        List<Map> employees = Arrays.asList(response.getBody());
        for (Map employee : employees) {
            System.out.println(employee);
        }

        // Create a new employee
        Map<String, Object> newEmployee = new HashMap<>(); // set the properties of the new employee
        Map createdEmployee = restTemplate.postForObject(BASE_URL + "/employees", newEmployee, Map.class);
        System.out.println(createdEmployee);

        // Get an employee by ID
        Long id = 1L; // replace with the ID of the employee you want to get
        Map employee = restTemplate.getForObject(BASE_URL + "/employees/" + id, Map.class);
        System.out.println(employee);
    }
}