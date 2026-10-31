document.querySelectorAll('a[href="#emailForm"]').forEach(link => {
  link.addEventListener('click', () => { document.getElementById('emailForm-email').focus({preventScroll:true}); });
});
document.querySelectorAll('form').forEach(form => {
  const button = form.querySelector('button');
  const label = button.querySelector('.btn-label');
  const status = document.getElementById(form.id + '-status');
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (button.disabled || !form.reportValidity()) return;
    if (form.elements._gotcha.value) return;
    button.disabled = true; label.textContent = 'Joining…'; status.textContent = '';
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(form.action, {method:'POST', headers:{Accept:'application/json','Content-Type':'application/json'},
        body:JSON.stringify({email:form.elements.email.value, _gotcha:'', consent:'SnowLens pilot email updates; age 18+; policy 2026-10-31'}),signal:controller.signal});
      if (!response.ok) throw new Error('Submission was not accepted');
      status.textContent = 'Thank you — your signup was received. We’ll email you with pilot updates; no access date is set.';
      form.reset(); label.textContent = 'Joined';
    } catch (error) {
      status.textContent = error.name === 'AbortError' ? 'We couldn’t confirm receipt in time. Please try again or email donald@getsnowlens.com.' : 'We couldn’t confirm your signup. Please try again or email donald@getsnowlens.com.';
      label.textContent = 'Try again'; button.disabled = false;
    } finally { clearTimeout(timeout); }
  });
});
