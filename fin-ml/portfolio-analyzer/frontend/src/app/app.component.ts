import { Component } from '@angular/core';
import { PortfolioAnalyzerUiComponent } from './portfolio-analyzer-ui/portfolio-analyzer-ui.component';
import { MatTabsModule } from '@angular/material/tabs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [PortfolioAnalyzerUiComponent, MatTabsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'portfolio-analyzer';
}
