import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class RuntimeConfigService {
  private config: any;

  constructor(private http: HttpClient) {}

  loadConfig(): Observable<void> {
    return this.http.get('/assets/runtime-config.json').pipe(
      map((config: any) => {
        this.config = config;
      })
    );
  }

  get baseUrl(): string {
    return this.config?.BASE_URL || '';
  }
}