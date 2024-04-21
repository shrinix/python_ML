import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatResponse } from './chatresponse';
import { HttpParams } from '@angular/common/http';
import { map } from 'rxjs/operators';
import { ChatMessage } from './chat.component';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  private baseURL = "http://localhost:8080";
  messages: ChatMessage[] = [];

  constructor(private httpClient: HttpClient) { }
  
  @Injectable({
    providedIn: 'root'
  })

  getAnswer(question: string): any {
    console.log ('question: '+question);
    const params = new HttpParams().set('userMessage', question);
    // const headers = new HttpHeaders()
    //   .set('Access-Control-Allow-Origin', '*')
    //   .set('Content-Type', 'application/json')
    //   .set('Origin', 'http://localhost:4200'); // replace with the actual origin of your client application

    return this.httpClient.get<ChatResponse>(`${this.baseURL}/generate?userMessage=${question}`);
  }

  addMessage(message: ChatMessage) {
    this.messages.push(message);
  }
}
