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
       return employeeService.findEmployeeById(employeeId);
    }

    @Tool("Get employee details by employee firstName for ShriniwasIyengarInc")
    public Employee findEmployeeByFirstName(String firstName, String lastName) {
        System.out.println("Invoking EmployeeTools.findEmployeeByFirstName: "+firstName);
        List<Employee> employees = employeeService.findAllEmployees();
        for (Employee employee : employees) {
            if (employee.getFirstName().equals(firstName) && employee.getLastName().equals(lastName)) {
                return employee;
            }
        }
        System.out.println("Employee not found with name: "+firstName+""+lastName);
        return null;
    }

    @Tool("Get employee details by employee email ID for ShriniwasIyengarInc")
    public Employee findEmployeeByEmailID(String emailID) {
        System.out.println("Invoking EmployeeTools.findEmployeeByLastName: "+emailID);
        List<Employee> employees = employeeService.findAllEmployees();
        for (Employee employee : employees) {
            if (employee.getLastName().equals(emailID)) {
                return employee;
            }
        }
        System.out.println("Employee not found with emailID: "+emailID);
        return null;
    }

    @Tool("Update or modify an existing employee of ShriniwasIyengarInc using first name or last name or email ID")
    public Employee updateEmployee(String matchStr, String updatedFieldName, String updatedFieldValue) {
        Employee matchedEmployee   = null;
        System.out.println("Invoking EmployeeTools.updateEmployee with match string: "+matchStr);
        List<Employee> employees = employeeService.findAllEmployees();
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
            return employeeService.updateEmployee(matchedEmployee.getId(), matchedEmployee);
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
        return employeeService.findAllEmployees();
    }
}
