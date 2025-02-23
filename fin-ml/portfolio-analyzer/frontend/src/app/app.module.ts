import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatTabsModule } from '@angular/material/tabs';
import { RouterModule } from '@angular/router'; // Import RouterModule
import { AppComponent } from './app.component';
import { PortfolioAnalyzerUiComponent } from './portfolio-analyzer-ui/portfolio-analyzer-ui.component';
import { routes } from './app.routes'; // Import the routes
import { HttpClientModule } from '@angular/common/http'

@NgModule({
  declarations: [
    // PortfolioAnalyzerUiComponent
  ],
  imports: [
    AppComponent,
    BrowserModule,
    BrowserAnimationsModule,
    HttpClientModule,
    MatTabsModule, // Ensure MatTabsModule is imported
    PortfolioAnalyzerUiComponent, // Import the standalone component
    RouterModule.forRoot(routes), // Configure the router
  ],
  providers: [],
  // bootstrap: [AppComponent]
})
export class AppModule { }