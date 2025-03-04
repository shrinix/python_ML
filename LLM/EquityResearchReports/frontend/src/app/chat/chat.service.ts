import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatResponse } from './chatresponse';
import { HttpParams } from '@angular/common/http';
import { map } from 'rxjs/operators';
import { ChatMessage } from './chat.component';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  private baseURL: string;

  constructor(private httpClient: HttpClient) {
    this.baseURL = environment.baseURL;
  }

  messages: ChatMessage[] = [];
  
  @Injectable({
    providedIn: 'root'
  })

  getAnswer(question: string): any {
    console.log ('question: '+question);
    const params = new HttpParams().set('userMessage', question);
    const headers = new HttpHeaders()
      //.set('Access-Control-Allow-Origin', '*')
      //.set('Content-Type', 'application/json')
      //.set('origin','http://52.14.147.215:4200/'); // replace with the actual origin of your client application

    return this.httpClient.get<ChatResponse>(`${this.baseURL}/generate`, { params, headers });
  }

  addMessage(message: ChatMessage) {
    this.messages.push(message);
  }
}
