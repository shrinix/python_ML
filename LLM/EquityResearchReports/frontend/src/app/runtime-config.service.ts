import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map, catchError, of } from 'rxjs';
import { environment } from '../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class RuntimeConfigService {
  private config: any;

  constructor(private http: HttpClient) {}

  loadConfig(): Observable<void> {
    return this.http.get('../assets/runtime-config.json').pipe(
      map((config: any) => {
        this.config = config;
      }),
      catchError(() => {
        console.log('Could not load runtime config, using environment variables');
        this.config = { 
          BASE_URL: environment.baseURL,
          SOURCES_URL: environment.sourcesURL 
        };
        return of(void 0);
      })
    );
  }

  get configLoaded(): boolean {
    return !!this.config;
  }
  get sourcesUrl(): string {
    return this.config?.SOURCES_URL || '';
  }

  get baseUrl(): string {
    return this.config?.BASE_URL || '';
  }
}