import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PortfolioAnalyzerUiComponent } from './portfolio-analyzer-ui.component';

describe('PortfolioAnalyzerUiComponent', () => {
  let component: PortfolioAnalyzerUiComponent;
  let fixture: ComponentFixture<PortfolioAnalyzerUiComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PortfolioAnalyzerUiComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PortfolioAnalyzerUiComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
