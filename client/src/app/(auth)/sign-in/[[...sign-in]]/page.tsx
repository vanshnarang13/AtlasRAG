import React from 'react'
import {SignIn} from '@clerk/nextjs'

function SignInPage() {
  return (
    <div className= "flex justify-center items-center min-h-screen bg-gray-50">
      <SignIn />
    </div>
  )
}

export default SignInPage;