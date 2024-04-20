import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChatResponse } from './chatresponse';
import { HttpParams } from '@angular/common/http';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  private baseURL = "http://localhost:8080";

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

  getJokeByTopic(topic: string): Observable<ChatResponse> {
    console.log (`${this.baseURL}/generate/joke/${topic}`);
    const headers = new HttpHeaders().set('Access-Control-Allow-Origin', '*')
    .set('Origin', 'http://localhost:4200'); // replace with the actual origin of your client application

    return this.httpClient.get<ChatResponse>(`${this.baseURL}/generate/joke/${topic}`, { headers });
  }

}
