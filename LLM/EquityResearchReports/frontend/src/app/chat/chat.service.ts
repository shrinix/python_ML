import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { ChatResponse } from './chatresponse';
import { HttpParams } from '@angular/common/http';
import { map } from 'rxjs/operators';
import { catchError } from 'rxjs/operators';
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
    return this.httpClient.get(`${this.baseURL}/companies`).pipe(
      catchError((error: any) => {
        console.error('Error fetching companies:', error);
        return throwError(error);
      })
    );
  }
  
  getAnswer(question: string): any {
    console.log ('question: '+question);
    const params = new HttpParams().set('userMessage', question);
    // const headers = new HttpHeaders()
      // .set('Access-Control-Allow-Origin', '*')
      //.set('Content-Type', 'application/json')
      //.set('origin','http://52.14.147.215:4200/'); // replace with the actual origin of your client application

    return this.httpClient.get<ChatResponse>(`${this.baseURL}/generate`, { params }).pipe(
      catchError((error: any) => {
        console.error('Error getting answer:', error);
        return throwError(error);
      })
    );
  }

  addMessage(message: ChatMessage) {
    this.messages.push(message);
  }

  // uploadFile(file: File, uploadedCompanyName: String): Observable<any> {
  //   const formData = new FormData();
  //   formData.append('file', file, file.name);
  //   // Add uploadedCompanyName as a query parameter
  //   const params = new HttpParams().set('company_name', uploadedCompanyName.toString());

  //   return this.httpClient.post(`${this.baseURL}/upload`, formData, { params });
  // }

  // New method to call the /generate_IA_report endpoint
  generateIAReport(company:string): Observable<any> {
    console.log ('company: '+company);
    const params = new HttpParams().set('company', company);
    return this.httpClient.get(`${this.baseURL}/generate_IA_report`, {params}).pipe(
      catchError((error: any) => {
      console.error('Error generating IA report:', error);
      return throwError(error);
      })
    );
  }
}
