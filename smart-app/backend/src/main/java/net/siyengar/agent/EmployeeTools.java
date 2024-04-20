package net.siyengar.agent;

import java.util.List;

import org.springframework.stereotype.Component;

import dev.langchain4j.agent.tool.Tool;
import net.siyengar.model.Employee;
import net.siyengar.service.EmployeeService;

@Component
public class EmployeeTools {
    //TODO: Can these methods be redirected to the EmployeeController?
   private final EmployeeService employeeService;

    public EmployeeTools(EmployeeService employeeService) {
        System.out.println("EmployeeTools constructor");     
        this.employeeService = employeeService;
    }

    @Tool("Get employee details by employee id for ShriniwasIyengarInc")
    public Employee findEmployeeById(long employeeId) {
        System.out.println("Invoking EmployeeTools.findEmployeeById: "+employeeId);
        // //Create a mock employee object
        // Employee
        // employee = new Employee();
        // employee.setId(1);
        // employee.setFirstName("John");
        // employee.setLastName("Doe");
        // employee.setEmailId("john.doe@abc.com");
        // return new Employee();
       return employeeService.findEmployeeById(employeeId);
    }

    // @Tool("Get employee details by employee firstName for ShriniwasIyengarInc")
    // public Employee findEmployeeByFirstName(String firstName) {
    //     System.out.println("Invoking EmployeeTools.findEmployeeByFirstName: "+firstName);
    //     // //Create a mock employee object
    //     // Employee
    //     // employee = new Employee();
    //     // employee.setId(1);
    //     // employee.setFirstName("John");
    //     // employee.setLastName("Doe");
    //     // employee.setEmailId("john.doe@abc.com");
    //     // return new Employee();
    //    return employeeService.findEmployeeById(employeeId);
    // }

    @Tool("Update or modify an existing employee of ShriniwasIyengarInc")
    public Employee updateEmployee(String newFirstName, long employeeID) {
        System.out.println("Invoking EmployeeTools.updateEmployee with Id: "+employeeID);
        Employee employee = findEmployeeById(employeeID);
        if (!(newFirstName.isEmpty()))
            employee.setFirstName(newFirstName);
        else {
            System.out.println("First Name cannot be empty");
            return null;
        }
        return employeeService.saveEmployee(employee);
    }


    @Tool("Add a new employee to ShriniwasIyengarInc")
    public Employee addEmployee(String firstName, String lastName, String emailID) {
        Employee employee = new Employee(firstName, lastName, emailID);
        System.out.println("Invoking EmployeeTools.addEmployee: "+employee);
        return employeeService.saveEmployee(employee);
    }

    @Tool("Delete an existing employee from ShriniwasIyengarInc")
    public void deleteEmployee(long employeeId) {
        System.out.println("Invoking EmployeeTools.deleteEmployee: "+employeeId);
        //employeeService.deleteEmployee(employeeId);
    }

    // @Tool("Finds an existing employee by email")
    // public CustomerRecord findCustomerByEmail(String email) {
    //     return employeeService.findCustomertByEmail(email);
    // }

    @Tool("Gets a list of all employees in ShriniwasIyengarInc")
    public List<Employee> findAllEmployees() {
        System.out.println("Invoking EmployeeTools.findAllEmployees");

        //Create a list of mock employee objects 
        // Employee employee1 = new Employee();
        // employee1.setId(1);
        // employee1.setFirstName("John");
        // employee1.setLastName("Doe");
        // employee1.setEmailId("");
        // Employee employee2 = new Employee();
        // employee2.setId(2);
        // employee2.setFirstName("Jane");
        // employee2.setLastName("Doe");
        // employee2.setEmailId("");
        // return List.of(employee1, employee2);
        
        return employeeService.findAllEmployees();
    }
}
