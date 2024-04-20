package net.siyengar.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import net.siyengar.model.Employee;

@Repository
public interface EmployeeRepository extends JpaRepository<Employee, Long>{

}
