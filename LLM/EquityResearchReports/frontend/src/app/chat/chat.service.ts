import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatResponse } from './chatresponse';
import { HttpParams } from '@angular/common/http';
import { map } from 'rxjs/operators';
import { ChatMessage } from './chat.component';
import { environment } from '../../environments/environment';
import { RuntimeConfigService } from '../runtime-config.service';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  private baseURL: string;

  constructor(private httpClient: HttpClient, private configService: RuntimeConfigService) {
    this.baseURL = configService.baseUrl;
    //if baseURL is null or configService is not initialized, use environment variable
    if (!this.baseURL) {
      this.baseURL = environment.baseURL;
    }
    console.log('baseURL is set to ' + this.baseURL);
  }

  messages: ChatMessage[] = [];
  
  @Injectable({
    providedIn: 'root'
  })

  getCompaniesList(): Observable<any> {
    return this.httpClient.get(`${this.baseURL}/companies`);
  }
  
  getAnswer(question: string): any {
    console.log ('question: '+question);
    const params = new HttpParams().set('userMessage', question);
    const headers = new HttpHeaders()
      // .set('Access-Control-Allow-Origin', '*')
      //.set('Content-Type', 'application/json')
      //.set('origin','http://52.14.147.215:4200/'); // replace with the actual origin of your client application

    return this.httpClient.get<ChatResponse>(`${this.baseURL}/generate`, { params, headers });
  }

  addMessage(message: ChatMessage) {
    this.messages.push(message);
  }

  uploadFile(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.httpClient.post(`${this.baseURL}/upload`, formData);
  }
}
