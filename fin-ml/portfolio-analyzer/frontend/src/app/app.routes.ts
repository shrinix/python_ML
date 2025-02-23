import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatTabsModule } from '@angular/material/tabs';
import { RouterModule } from '@angular/router'; // Import RouterModule
import { AppComponent } from './app.component';
import { PortfolioAnalyzerUiComponent } from './portfolio-analyzer-ui/portfolio-analyzer-ui.component';
import { Routes } from '@angular/router';

@NgModule({
  declarations: [
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    MatTabsModule,
    PortfolioAnalyzerUiComponent,
    AppComponent,
  ],
  providers: [],
  // bootstrap: [AppComponent]
})
export class AppModule { }

export const routes: Routes = [
  { path: 'portfolio-analyzer', component: PortfolioAnalyzerUiComponent }, // Define the route
  { path: '', redirectTo: '/portfolio-analyzer', pathMatch: 'full' } // Redirect to the component by default
];