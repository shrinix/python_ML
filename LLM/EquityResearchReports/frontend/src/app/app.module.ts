import { BrowserModule } from '@angular/platform-browser';
import { NgModule,APP_INITIALIZER } from '@angular/core';
import { HttpClientModule } from '@angular/common/http'
import { AppRoutingModule } from './app-routing.module';
import { FormsModule} from '@angular/forms';
import { ChatComponent } from './chat/chat.component';
import { AppComponent } from './app.component';
import { Routes, RouterModule } from '@angular/router';
import { RuntimeConfigService } from './runtime-config.service';

export function initializeApp(runtimeConfigService: RuntimeConfigService) {
  return (): Promise<void> => runtimeConfigService.loadConfig().toPromise();
}

const routes: Routes = [
  {path: '', redirectTo: 'chat', pathMatch: 'full'},
  {path: 'chat', component: ChatComponent },
  {path: 'generate-answer/:question', component: ChatComponent }
];

@NgModule({
  declarations: [
    ChatComponent,
    AppComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    RouterModule.forRoot(routes)
  ],
  providers: [
    RuntimeConfigService,
    {
      provide: APP_INITIALIZER,
      useFactory: initializeApp,
      deps: [RuntimeConfigService],
      multi: true
    }
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
