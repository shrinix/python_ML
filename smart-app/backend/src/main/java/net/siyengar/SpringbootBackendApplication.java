package net.siyengar;

import static java.time.Duration.ofSeconds;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@EnableJpaRepositories("net.siyengar.repository")
@SpringBootApplication(scanBasePackages={"net.siyengar.controller", "net.siyengar.service", "net.siyengar.repository", "net.siyengar.exception",
  "net.siyengar.model", "net.siyengar.agent", "net.siyengar.config"})
public class SpringbootBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(SpringbootBackendApplication.class, args);
	}
}
