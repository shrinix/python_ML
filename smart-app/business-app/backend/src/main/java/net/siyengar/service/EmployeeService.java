package net.siyengar.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import net.siyengar.exception.ResourceNotFoundException;
import net.siyengar.model.Employee;
import net.siyengar.repository.EmployeeRepository;

@Service
public class EmployeeService {
    
    @Autowired
	private EmployeeRepository employeeRepository;

    public EmployeeService(EmployeeRepository employeeRepository) {
        this.employeeRepository = employeeRepository;
    }

    public List<Employee> findAllEmployees() {
        return employeeRepository.findAll();
    }

    public Employee findEmployeeById(long employeeId) {
        return employeeRepository.findById(employeeId)
            .orElseThrow(() -> new ResourceNotFoundException("Employee not exist with id :" + employeeId));
    }

    // public Employee findEmployeeByEmailId(String emailId) {
    //     return employeeRepository.findByEmailId(emailId)
    //         .orElseThrow(() -> new ResourceNotFoundException("Employee not exist with email id :" + emailId));
    // }

    public Employee saveEmployee(Employee employee) {
        return employeeRepository.save(employee);
    }

    public void deleteEmployee(long employeeId) {
        Employee employee = employeeRepository.findById(employeeId)
				.orElseThrow(() -> new ResourceNotFoundException("Employee not exist with id :" + employeeId));
		
        employeeRepository.deleteById(employeeId);
    }

    public Employee updateEmployee(long id, Employee employeeDetails) {
        Employee employee = employeeRepository.findById(id)
        .orElseThrow(() -> new ResourceNotFoundException("Employee not exist with id :" + id));

        employee.setFirstName(employeeDetails.getFirstName());
        employee.setLastName(employeeDetails.getLastName());
        employee.setEmailId(employeeDetails.getEmailId());

        Employee updatedEmployee = employeeRepository.save(employee);

        return updatedEmployee;
    }
}
