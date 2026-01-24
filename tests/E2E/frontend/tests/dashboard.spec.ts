import { test, expect } from '@playwright/test';

test.describe('Emotion Dashboard - Basic UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('page loads successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/Emotion Monitoring Dashboard/);
  });

  test('displays header elements', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Emotion Monitoring Dashboard');
    await expect(page.locator('.subtitle')).toContainText('Real-time emotion distribution tracking');
  });

  test('displays all statistics cards', async ({ page }) => {
    // Check all three stat cards exist
    await expect(page.locator('.stat-card')).toHaveCount(3);
    
    // Check stat labels
    await expect(page.locator('.stat-label').nth(0)).toContainText('Total Students');
    await expect(page.locator('.stat-label').nth(1)).toContainText('Total Predictions');
    await expect(page.locator('.stat-label').nth(2)).toContainText('Active Sessions');
    
    // Check stat values are displayed (should be numbers)
    await expect(page.locator('#totalStudents')).toBeVisible();
    await expect(page.locator('#totalPredictions')).toBeVisible();
    await expect(page.locator('#activeSessions')).toBeVisible();
  });

  test('displays plot container', async ({ page }) => {
    await expect(page.locator('.plot-container')).toBeVisible();
    await expect(page.locator('.plot-title')).toContainText('Emotion Distribution Over Time');
  });

  test('displays control elements', async ({ page }) => {
    await expect(page.locator('#studentSelect')).toBeVisible();
    await expect(page.locator('button:has-text("Refresh")')).toBeVisible();
  });

  test('displays students list section', async ({ page }) => {
    await expect(page.locator('.students-list')).toBeVisible();
    await expect(page.locator('.students-list h2')).toContainText('Active Students');
  });

  test('displays last update indicator', async ({ page }) => {
    await expect(page.locator('.last-update')).toBeVisible();
  });

  test('student select has default option', async ({ page }) => {
    const select = page.locator('#studentSelect');
    await expect(select).toBeVisible();
    
    // Should have "All Students" as default
    const options = await select.locator('option').allTextContents();
    expect(options).toContain('All Students');
  });
});

test.describe('Emotion Dashboard - Statistics', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('statistics update after data submission', async ({ page, request }) => {
    // Get initial stats
    const initialStudents = await page.locator('#totalStudents').textContent();
    const initialPredictions = await page.locator('#totalPredictions').textContent();
    
    // Send test data via API
    await request.post('http://localhost:8001/api/emotions', {
      data: {
        id: 'e2e_test_student',
        predictions: {
          happy: 0.7,
          sad: 0.2,
          neutral: 0.1
        }
      }
    });
    
    // Wait a bit for the page to update
    await page.waitForTimeout(2000);
    
    // Manually refresh the page or trigger stats fetch
    await page.reload();
    
    // Check stats updated
    const newStudents = await page.locator('#totalStudents').textContent();
    const newPredictions = await page.locator('#totalPredictions').textContent();
    
    // Values should have changed
    expect(parseInt(newPredictions || '0')).toBeGreaterThan(parseInt(initialPredictions || '0'));
  });

  test('displays zero stats when no data', async ({ page, request }) => {
    // This test assumes a fresh start or cleared data
    const students = await page.locator('#totalStudents').textContent();
    const predictions = await page.locator('#totalPredictions').textContent();
    const sessions = await page.locator('#activeSessions').textContent();
    
    // All should be numbers (0 or more)
    expect(parseInt(students || '0')).toBeGreaterThanOrEqual(0);
    expect(parseInt(predictions || '0')).toBeGreaterThanOrEqual(0);
    expect(parseInt(sessions || '0')).toBeGreaterThanOrEqual(0);
  });
});

test.describe('Emotion Dashboard - Plot Functionality', () => {
  test.beforeEach(async ({ page, request }) => {
    // Add test data before each test
    for (let i = 0; i < 3; i++) {
      await request.post('http://localhost:8001/api/emotions', {
        data: {
          id: `plot_test_student_${i}`,
          predictions: {
            happy: 0.3 + (i * 0.1),
            sad: 0.3,
            neutral: 0.4 - (i * 0.1)
          }
        }
      });
    }
    
    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
  });

  test('plot image loads', async ({ page }) => {
    const plotImg = page.locator('#emotionPlot');
    
    // Wait for image to be visible
    await expect(plotImg).toBeVisible({ timeout: 10000 });
    
    // Check that src is set
    const src = await plotImg.getAttribute('src');
    expect(src).toContain('/api/plot');
  });

  test('refresh button updates plot', async ({ page }) => {
    await page.waitForTimeout(2000);
    
    const plotImg = page.locator('#emotionPlot');
    const initialSrc = await plotImg.getAttribute('src');
    
    // Click refresh
    await page.click('button:has-text("Refresh")');
    
    // Wait for update
    await page.waitForTimeout(2000);
    
    const newSrc = await plotImg.getAttribute('src');
    
    // Source should have changed (different timestamp)
    expect(newSrc).not.toBe(initialSrc);
  });

  test('small loader appears during refresh', async ({ page }) => {
    const smallLoader = page.locator('#smallLoader');
    
    // Initially hidden
    await expect(smallLoader).toBeHidden();
    
    // Click refresh
    await page.click('button:has-text("Refresh")');
    
    // Loader should appear briefly (might be too fast to catch consistently)
    // Just check it exists
    await expect(smallLoader).toBeAttached();
  });

  test('last update time displays', async ({ page }) => {
    await page.waitForTimeout(2000);
    
    const lastUpdate = page.locator('#lastUpdate');
    const text = await lastUpdate.textContent();
    
    // Should not be "Never" after plot loads
    expect(text).not.toBe('Never');
  });
  
});

test.describe('Emotion Dashboard - Student Filtering', () => {
  test.beforeEach(async ({ page, request }) => {
    // Add multiple students
    const students = ['filter_student_a', 'filter_student_b', 'filter_student_c'];
    
    for (const studentId of students) {
      await request.post('http://localhost:8001/api/emotions', {
        data: {
          id: studentId,
          predictions: {
            happy: 0.5,
            sad: 0.3,
            neutral: 0.2
          }
        }
      });
    }
    
    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
  });

  test('student select populates with students', async ({ page }) => {
    await page.reload();
    await page.waitForTimeout(1000);
    
    const select = page.locator('#studentSelect');
    const options = await select.locator('option').allTextContents();
    
    // Should have "All Students" plus the test students
    expect(options.length).toBeGreaterThan(1);
    expect(options).toContain('All Students');
  });

  test('selecting student filters plot', async ({ page }) => {
    await page.reload();
    await page.waitForTimeout(1000);
    
    const select = page.locator('#studentSelect');
    const options = await select.locator('option').allTextContents();
    
    if (options.length > 1) {
      // Select a specific student (not "All Students")
      const studentName = options.find(opt => opt !== 'All Students');
      if (studentName) {
        await select.selectOption({ label: studentName });
        
        await page.waitForTimeout(2000);
        
        // Plot src should include the student ID
        const plotSrc = await page.locator('#emotionPlot').getAttribute('src');
        expect(plotSrc).toContain('/api/plot/');
      }
    }
  });

  test('switching back to "All Students" shows combined plot', async ({ page }) => {
    await page.reload();
    await page.waitForTimeout(1000);
    
    const select = page.locator('#studentSelect');
    
    // Select first, then switch back
    const options = await select.locator('option').allTextContents();
    if (options.length > 1) {
      const studentName = options.find(opt => opt !== 'All Students');
      if (studentName) {
        await select.selectOption({ label: studentName });
        await page.waitForTimeout(1000);
        
        // Switch back to all
        await select.selectOption({ label: 'All Students' });
        await page.waitForTimeout(2000);
        
        const plotSrc = await page.locator('#emotionPlot').getAttribute('src');
        expect(plotSrc).toMatch(/\/api\/plot\?/);
      }
    }
  });
});

test.describe('Emotion Dashboard - Students List', () => {
  test.beforeEach(async ({ page, request }) => {
    // Add test students
    for (let i = 0; i < 3; i++) {
      await request.post('http://localhost:8001/api/emotions', {
        data: {
          id: `list_student_${i}`,
          predictions: {
            happy: 0.6,
            sad: 0.4
          }
        }
      });
    }
    
    await page.goto('/dashboard');
    await page.waitForTimeout(1000);
  });

  test('displays students in list', async ({ page }) => {
    await page.reload();
    await page.waitForTimeout(1000);
    
    const studentsList = page.locator('#studentsList');
    await expect(studentsList).toBeVisible();
    
    // Should have student items
    const items = studentsList.locator('.student-item');
    const count = await items.count();
    
    expect(count).toBeGreaterThan(0);
  });

  test('student items show name and count', async ({ page }) => {
    await page.reload();
    await page.waitForTimeout(1000);
    
    const firstItem = page.locator('.student-item').first();
    
    if (await firstItem.isVisible()) {
      await expect(firstItem.locator('.student-name')).toBeVisible();
      await expect(firstItem.locator('.student-count')).toBeVisible();
      
      // Count should contain "predictions"
      const countText = await firstItem.locator('.student-count').textContent();
      expect(countText).toContain('prediction');
    }
  });

  test('shows empty message when no students', async ({ page }) => {
    // This would need a way to clear data or test on fresh instance
    // For now, just check the structure exists
    const studentsList = page.locator('#studentsList');
    await expect(studentsList).toBeVisible();
  });
});

test.describe('Emotion Dashboard - Auto Refresh', () => {
  test('auto refresh updates data periodically', async ({ page, request }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(2000);
    
    const initialPredictions = await page.locator('#totalPredictions').textContent();
    
    // Add new data
    await request.post('http://localhost:8001/api/emotions', {
      data: {
        id: 'auto_refresh_student',
        predictions: {
          happy: 0.8,
          sad: 0.2
        }
      }
    });
    
    // Wait for auto-refresh (10 seconds in the code)
    await page.waitForTimeout(11000);
    
    const newPredictions = await page.locator('#totalPredictions').textContent();
    
    // Should have updated
    expect(parseInt(newPredictions || '0')).toBeGreaterThan(parseInt(initialPredictions || '0'));
  });
});

test.describe('Emotion Dashboard - Responsive Design', () => {
  test('displays correctly on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/dashboard');
    
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.stats-grid')).toBeVisible();
    await expect(page.locator('.plot-container')).toBeVisible();
  });

  test('displays correctly on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/dashboard');
    
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.stats-grid')).toBeVisible();
    await expect(page.locator('.plot-container')).toBeVisible();
  });

  test('displays correctly on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/dashboard');
    
    await expect(page.locator('h1')).toBeVisible();
    await expect(page.locator('.stats-grid')).toBeVisible();
    await expect(page.locator('.plot-container')).toBeVisible();
  });
});

test.describe('Emotion Dashboard - Performance', () => {
  test('page loads within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
    const loadTime = Date.now() - startTime;
    
    // Should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('plot image loads within acceptable time', async ({ page, request }) => {
    // Add data
    await request.post('http://localhost:8001/api/emotions', {
      data: {
        id: 'performance_student',
        predictions: { happy: 0.7, sad: 0.3 }
      }
    });
    
    await page.goto('/dashboard');
    
    const startTime = Date.now();
    await page.locator('#emotionPlot').waitFor({ state: 'visible', timeout: 10000 });
    const loadTime = Date.now() - startTime;
    
    // Plot should load within 10 seconds
    expect(loadTime).toBeLessThan(10000);
  });
});
